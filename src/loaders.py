from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import Resize

from src.augments import Augmenter
from src.config import Config
from src.constants import N_ENSEMBLES, REPRO_DIR, SHUFFLE_SEED, VAL_SIZE
from src.enumerables import Phase, TrainingSubset, VisionBinaryDataset, VisionDataset


def get_shuffle_idxs(n_indices: int, size: int) -> ndarray:
    """
    Parameters
    ----------
    n_indices: int
        Number of indices to generate. Should be equal to number of ensembles.

    size: int
        Size of the array to be shuffled.
    """
    outfile = REPRO_DIR / f"shuffle_idx_{n_indices}x{size}.npy"
    if outfile.exists():
        return np.load(outfile, allow_pickle=False, fix_imports=False)
    ss = np.random.SeedSequence(entropy=SHUFFLE_SEED)
    seeds = ss.spawn(n_indices)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    idxs = np.staack([rng.permutation(size) for rng in rngs])
    np.save(outfile, idxs, allow_pickle=False, fix_imports=False)
    print(f"Saved shuffle indices to {outfile}.")
    return idxs


def get_boot_idxs(n_indices: int, size: int) -> ndarray:
    """
    Parameters
    ----------
    n_indices: int
        Number of indices to generate. Should be equal to number of ensembles.

    size: int
        Size of the bootstrap resample. Should be size of base-training set.
    """
    outfile = REPRO_DIR / f"boot_idx_{n_indices}x{size}.npy"
    if outfile.exists():
        return np.load(outfile, allow_pickle=False, fix_imports=False)
    ss = np.random.SeedSequence(entropy=SHUFFLE_SEED)
    seeds = ss.spawn(n_indices)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    idxs = np.stack([rng.integers(low=0, high=size, size=size) for rng in rngs])
    np.save(outfile, idxs, allow_pickle=False, fix_imports=False)
    print(f"Saved bootstrap indices to {outfile}.")
    return idxs


class NormedDataset(TensorDataset):
    def __init__(
        self,
        *tensors: Tensor,
        train_means: ndarray,
        train_sds: ndarray,
        augment: bool = False,
        resize: int | None = None,
    ) -> None:
        super().__init__(*tensors)
        self.augment = augment
        self.resize = resize
        self.means = train_means
        self.sds = train_sds
        if augment:
            self.augmenter = Augmenter()
        if self.resize is not None:
            self.resizer = Resize(self.resize)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x, y = super().__getitem__(index)
        if self.augment:
            x = self.augmenter(x)
        if self.resize is not None:
            x = self.resizer(x)
        # This follows the order of TA paper
        x = (x - self.means) / self.sds
        return x, y


def to_BCHW(X: ndarray) -> ndarray:
    # make X.shape be (B, C, H, W) always
    if X.ndim == 3:
        X = np.stack([X, X, X], axis=1)
    else:
        X = X.transpose(0, 3, 1, 2)
    return X


def get_train_val_splits(
    X: ndarray, y: ndarray, ensemble_idx: int
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    # select base-learner subset
    kf = StratifiedKFold(n_splits=50, shuffle=False)
    idx_train = list(kf.split(y, y))[ensemble_idx][0]
    X = X[idx_train]
    y = y[idx_train]
    idx_boot = get_boot_idxs(n_indices=N_ENSEMBLES, size=X.shape[0])[ensemble_idx]
    X = X[idx_boot]
    y = y[idx_boot]

    # split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=SHUFFLE_SEED, shuffle=False
    )
    return X_tr, X_val, y_tr, y_val


def vision_datasets(config: Config) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """"""
    kind = config.vision_dataset
    binary = config.binary
    subset = config.subset  # TODO
    ensemble_idx = config.ensemble_idx
    if (ensemble_idx is None) and (subset is TrainingSubset.Boot):
        raise ValueError(
            "Misconfiguration. Bootstrap resamples are only used for training "
            "ensemble base learners."
        )

    X = kind.x_train()
    y = kind.y_train()
    X_test = kind.x_train()
    y_test = kind.y_train()

    X = to_BCHW(X)
    X_test = to_BCHW(X_test)

    if binary:
        l1, l2 = kind.binary().classes()  # binary labels
        idx = (y == l1) | (y == l2)
        X, y = np.copy(X[idx]), np.copy(y[idx])  # copy defragments
        X_test, y_test = np.copy(X_test[idx]), np.copy(y_test[idx])

    X_full = np.copy(X)
    y_full = np.copy(y)

    # It probably makes sense to use the full set of training data for statistics,
    # as the ensemble still collectively trains on most of the training data, and
    # the unused training data for each base learner is not used in testing or any
    # where else. So collect stats before train/val splits.
    train_means = np.mean(X, axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)
    train_sds = np.std(X, axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)
    if ensemble_idx is None:  # subset is TrainingSubset.Full
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=VAL_SIZE, random_state=SHUFFLE_SEED, shuffle=False
        )
    else:
        X_tr, X_val, y_tr, y_val = get_train_val_splits(X, y, ensemble_idx=ensemble_idx)
    norm_args = dict(train_means=train_means, train_sds=train_sds)
    return (
        NormedDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr), augment=True, **norm_args
        ),
        NormedDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), **norm_args),
        NormedDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), **norm_args),
        NormedDataset(torch.from_numpy(X_full), torch.from_numpy(y_full), **norm_args),
    )


def vision_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    train, val, test, full = vision_datasets(config)
    shared_args: Mapping = dict(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    train_loader = DataLoader(train, shuffle=True, **shared_args)
    val_loader = DataLoader(val, shuffle=False, **shared_args)
    test_loader = DataLoader(test, shuffle=False, **shared_args)  # type: ignore
    train_boot_loader = DataLoader(train, shuffle=False, **shared_args)
    train_full_loader = DataLoader(full, shuffle=False, **shared_args)
    return train_loader, val_loader, test_loader, train_boot_loader, train_full_loader


if __name__ == "__main__":
    for ds in VisionDataset:
        config = Config.from_args(f"--data={ds.value}")[0]
        vision_loaders(config)