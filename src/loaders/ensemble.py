from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
from typing import Mapping

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import Resize

from src.augments import Augmenter
from src.config import Config
from src.constants import (
    CC_LOGS,
    LOG_ROOT_DIR,
    N_ENSEMBLES,
    ON_CCANADA,
    REPRO_DIR,
    SHUFFLE_SEED,
    VAL_SIZE,
)
from src.enumerables import (
    FinalEvalPhase,
    Phase,
    TrainingSubset,
    VisionBinaryDataset,
    VisionDataset,
)
from src.loaders.loaders import get_boot_idxs
from src.loaders.preds import consolidate_preds

LOG_DIR = LOG_ROOT_DIR if ON_CCANADA else CC_LOGS / "logs"
BOOT_DIR = LOG_DIR / "base-train/boot"


class EnsembleTrain(Dataset):
    """
    Notes
    -----
    There are two ways to design a stacked learner here. Each ensemble yields,
    for each training sample, logits of shape (num_classes=C,). If we have E
    ensembles and N training samples, then the base training data X is shape
    (E, N, C).

    For each sample, and for each ensemble, there is a correct class label.
    That is, there is a tensor of correct class labels, `targs` with shape
    (E, N). Because of subsetting, we CANNOT say that e.g. Ensemble `e1`, sample
    `i` has the same correct label as ensemble `e2` sample `i`, that is

        targs[e1, i] != targs[e2, i]

    IN GENERAL (i.e. if using bootstrap samples). However, on the full training
    data, this IS true.

    # Bootstrap / Pooling Approach

    In the bootstrap case, we can thus treat the entire set of ensemble logits
    as a single pool to draw from, e.g. we have

        X_train = X.reshape(E*N, C)
        y_train = targs.ravel()  # shape is (E*N,)

    and return as (X, y) pairs (X_train[i], y_train[i]). In most cases one would
    obtain the correct prediction by just argmax'ing X_train[i], but one hopes
    that in this case the machine learns not argmax, but somehow something other
    than argmax when this would in fact yield the wrong result.

    # Full-Train Approach

    In this case, we DO know targs[e1, i] == targs[e2, i] for any e1, e2, and
    so we can write:

        X_train = X  # X_train.shape == (E, N, C)
        y_train = targs[0]  # shape is (N,)

    and return as (X, y) pairs (X_train[:, i, :], y_train[i]), such that X.shape
    is (E, C).

    Now the problem with this approach is that the meta-learner can overfit ensemble
    ordering. This may or may not be good. E.g. if some base-learner was particularly
    bad, perhaps we want to weight its contributions to zero, but if some base learners
    were better/worse just by coincedence, we risk memorizing spurious patterns.

    Or, put another way, it is important to ask if the predictions from an ensemble
    should be *insensitive to the orderings of that ensemble*. A meta-learner sensitive
    to the ensemble ordering can learn to exclude certain base-learner by virtue of
    their position (index) alone, and this might be good if there is a particularly
    bad base learner. However, if this is not the case, then one might prefer instead
    that the machine recognizes rather patterns only in the *set* of predictions for
    certain samples. In this case we do not want sensitivity to ordering.

    We could implement insensitivity to ordering at the architecture level (e.g. PointNet),
    or we could just allow shuffling at the data loading level, i.e. where we shuffle
    along the ensemble axis.
    """

    def __init__(
        self,
        config: Config,
        threshold: float | None,
        source: FinalEvalPhase,
        pooled_ensembles: bool,
        shuffled: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.source = source
        self.threshold = threshold
        self.pooled = pooled_ensembles
        self.shuffle = shuffled
        if source is FinalEvalPhase.Test or source is FinalEvalPhase.Val:
            raise ValueError("Cannot train ensemble on testing or validation data!")
        if self.pooled and self.shuffle:
            raise ValueError(
                "Shuffling does not make sense when pooling ensemble logits."
            )
        if (not self.pooled) and (source is not FinalEvalPhase.FullTrain):
            raise ValueError(
                "If not pooling logits for the meta-learner, then all training "
                "predictions must be aligned, i.e. on the same samples. This is "
                "only possible if using the predictions on the full training "
                "data, i.e. `source=FinalEvalPhase.FullTrain`."
            )

        self.preds: ndarray  # shape (N_ensembles, N_samples_full_train, N_classes)
        self.targs: ndarray  # shape(N_ensembles, N_samples_full_train)
        self.preds, self.targs, self.idxs = consolidate_preds(
            config.vision_dataset, phase=source, threshold=threshold
        )
        self.num_classes = self.preds.shape[-1]

        nan_ensembles = np.unique(np.where(np.isnan(self.preds))[0]).ravel().tolist()
        self.preds = np.delete(self.preds, nan_ensembles, axis=0)
        self.targs = np.delete(self.targs, nan_ensembles, axis=0)

        if self.pooled:
            # reshape as in Notes
            length = np.prod(self.preds.shape[:-1])
            self.X = self.preds.reshape(length, self.num_classes)  # (N*E, C)
            self.y = self.targs.reshape(length)  # (N*E,)
        else:
            # shape is (E, N, C) to start
            self.X = np.copy(self.preds.transpose(1, 0, 2))  # now (N, E, C)
            self.y = np.copy(self.targs[0, :])  # now (N,)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """We just want to make a prediction from the raw logits of N_ENSEMBLE
        base-learner predictions.

        Returns
        -------
        x: Tensor
            If `self.pooled=True`, Tensor of shape (N_classes,), i.e. the raw
            logits for sample `index` from the full available training data,
            for some ensemble.

            If `self.pooled=False`, then a Tensor of shape (N_ensembles,
            N_classes), i.e. the raw logits for each ensemble and each class of
            sample `index` from the full available training data. If
            `self.shuffle=True`, then `x` is shuffled along the ensemble
            dimension, to precent learning or ordering information.

        y: Tensor
            If `self.pooled`, Tensor of shape (1,), which is the value of the true class for
            the sample at `index`.
        """
        x = torch.from_numpy(self.X[index])
        y = torch.tensor(self.y[index])
        if self.shuffle:
            idx = torch.randperm(x.shape[0])
            x = x[idx]
        return x.ravel().contiguous(), y

    def __len__(self) -> int:
        return len(self.y)


class EnsembleTest(Dataset):
    def __init__(
        self,
        config: Config,
        threshold: float | None,
        pooled_ensembles: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.source = FinalEvalPhase.Test
        self.threshold = threshold
        self.pooled = pooled_ensembles
        self.shuffle = False
        if self.pooled and self.shuffle:
            raise ValueError(
                "Shuffling does not make sense when pooling ensemble logits."
            )

        self.preds: ndarray  # shape (N_ensembles, N_samples_full_train, N_classes)
        self.targs: ndarray  # shape(N_ensembles, N_samples_full_train)
        self.preds, self.targs, self.idxs = consolidate_preds(
            config.vision_dataset, phase=self.source, threshold=threshold
        )
        self.num_classes = self.preds.shape[-1]

        nan_ensembles = np.unique(np.where(np.isnan(self.preds))[0]).ravel().tolist()
        self.preds = np.delete(self.preds, nan_ensembles, axis=0)
        self.targs = np.delete(self.targs, nan_ensembles, axis=0)

        if self.pooled:
            # reshape as in Notes
            length = np.prod(self.preds.shape[:-1])
            self.X = self.preds.reshape(length, self.num_classes)  # (N*E, C)
            self.y = self.targs.reshape(length)  # (N*E,)
        else:
            # shape is (E, N, C) to start
            self.X = np.copy(self.preds.transpose(1, 0, 2))  # now (N, E, C)
            self.y = np.copy(self.targs[0, :])  # now (N,)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """We just want to make a prediction from the raw logits of N_ENSEMBLE
        base-learner predictions.

        Returns
        -------
        x: Tensor
            If `self.pooled=True`, Tensor of shape (N_classes,), i.e. the raw
            logits for sample `index` from the full available training data,
            for some ensemble.

            If `self.pooled=False`, then a Tensor of shape (N_ensembles,
            N_classes), i.e. the raw logits for each ensemble and each class of
            sample `index` from the full available training data. If
            `self.shuffle=True`, then `x` is shuffled along the ensemble
            dimension, to precent learning or ordering information.

        y: Tensor
            If `self.pooled`, Tensor of shape (1,), which is the value of the true class for
            the sample at `index`.
        """
        x = torch.from_numpy(self.X[index])
        y = torch.tensor(self.y[index])
        if self.shuffle:
            idx = torch.randperm(x.shape[0])
            x = x[idx]
        return x.ravel().contiguous(), y

    def __len__(self) -> int:
        return len(self.y)

    def pred_shape(self) -> tuple[int, ...]:
        if self.pooled:
            return (self.config.num_classes,)
        else:
            return (self.preds.shape[0], self.config.num_classes)


def ensemble_loaders(
    config: Config, threshold: float | None, pooled_ensembles: bool, shuffled: bool
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    all_train = EnsembleTrain(
        config,
        source=FinalEvalPhase.FullTrain,
        threshold=threshold,
        pooled_ensembles=pooled_ensembles,
        shuffled=shuffled,
    )
    test_data = EnsembleTest(
        config=config, threshold=threshold, pooled_ensembles=pooled_ensembles
    )
    train_size = int(len(all_train) * 0.9)
    # train_size = int(len(all_train) * 0.95)
    val_size = len(all_train) - train_size
    train_data, val_data = random_split(all_train, lengths=(train_size, val_size))
    args = dict(batch_size=config.batch_size, num_workers=config.num_workers)
    train = DataLoader(train_data, shuffle=True, **args)
    val = DataLoader(val_data, shuffle=False, **args)
    test = DataLoader(test_data, shuffle=False, **args)
    in_channels = int(np.prod(test_data.pred_shape()))
    return train, val, test, in_channels


def ml_data(dataset: VisionDataset) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train, y_train, train_idxs = consolidate_preds(  # X_train.shape (E, N, C)
        dataset, phase=FinalEvalPhase.FullTrain
    )
    X_test, y_test, test_idxs = consolidate_preds(
        dataset,
        phase=FinalEvalPhase.Test,
    )
    n_ensembles, n_samples, n_classes = X_train.shape
    X_train = X_train.transpose(1, 0, 2).reshape(-1, n_ensembles * n_classes)
    X_test = X_test.transpose(1, 0, 2).reshape(-1, n_ensembles * n_classes)
    return X_train, y_train[0], X_test, y_test[0]