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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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

LOG_DIR = LOG_ROOT_DIR if ON_CCANADA else CC_LOGS / "logs"
BOOT_DIR = LOG_DIR / "base-train/boot"

"""
NOTE: There is an issue with train_boot_preds because of an oversight in `vision_loaders`, namely,
the predictions here were produced using augments (ugh). However, train_boot_preds are just a sub-
set of `train_full_preds` and ought to be re-selectable from them using
`src.loaders.loaders.get_boot_idxs` or even just `src.loaders.loaders.get_train_val_splits`.
"""


def get_boot_samples(
    y: ndarray, preds: ndarray, targs: ndarray, ensemble_idx: int
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Parameters
    ----------
    y: ndarray
        Original train labels. Needed to replicate subsetting.

    preds: ndarray
        Predictions (raw linear outputs) to subset to bootstrap sample.

    targs: ndarray
        Targets (true class labels) to subset to bootstrap sample.

    Notes
    -----
    Boot preds were produced from 50-fold fold `ensemble_idx` bootstrap sample.
    To get from full preds to repeat boot preds, just need to replicate that
    process on the preds. Full preds did NOT have validation samples set aside.
    So length of full train preds is same as `len(y)` for `y` passed in to this
    function.
    """
    # remove the 2% from the 50-fold
    kf = StratifiedKFold(n_splits=N_ENSEMBLES, shuffle=False)
    idx_train = list(kf.split(y, y))[ensemble_idx][0]
    preds = preds[idx_train]
    targs = targs[idx_train]

    N = len(y)
    N_BOOT = (N - (N / N_ENSEMBLES)) * (1 - VAL_SIZE)
    # not
    idx_boot = get_boot_idxs(n_indices=N_ENSEMBLES, size=preds.shape[0])[ensemble_idx]
    preds = preds[idx_boot]
    targs = targs[idx_boot]

    # split train into train/val
    preds_tr, preds_val, targs_tr, targs_val = train_test_split(
        preds, targs, test_size=VAL_SIZE, random_state=SHUFFLE_SEED, shuffle=False
    )
    return preds_tr, targs_tr


def consolidate_preds(
    dataset: VisionDataset, phase: FinalEvalPhase
) -> tuple[ndarray, ndarray, ndarray]:
    args = dict(allow_pickle=False, fix_imports=False)
    predsfile = REPRO_DIR / f"{dataset.value}_{phase.value}_consolidated_preds.npy"
    targsfile = REPRO_DIR / f"{dataset.value}_{phase.value}_consolidated_targs.npy"
    idxsfile = REPRO_DIR / f"{dataset.value}_{phase.value}_consolidated_ensemble_idx.npy"
    if predsfile.exists() and targsfile.exists() and idxsfile.exists():
        return (
            np.load(predsfile, **args),
            np.load(targsfile, **args),
            np.load(idxsfile, **args),
        )

    ds = dataset
    is_boot = phase is FinalEvalPhase.BootTrain
    # we have to handle the bad (augmented) boot predictions
    phase = FinalEvalPhase.FullTrain if is_boot else phase

    configs = sorted(
        filter(lambda p: f"{ds.value}/" in str(p), BOOT_DIR.rglob("config.json"))
    )
    pred_dirs = [conf.parent.parent / f"{phase.value}_preds" for conf in configs]
    pred_files = [list(path.glob("*preds_epoch*.npy"))[0] for path in pred_dirs]
    targ_files = [list(path.glob("*labels_epoch*.npy"))[0] for path in pred_dirs]

    preds = [np.load(path) for path in pred_files]
    targs = [np.load(path) for path in targ_files]
    ensemble_idxs = [int(re.search(r"idx_(\d+)", str(p))[1]) for p in pred_dirs]
    if not is_boot:
        preds = np.stack(preds)
        targs = np.stack(targs)
        idxs = np.stack(ensemble_idxs)
        np.save(predsfile, preds, **args)
        np.save(targsfile, targs, **args)
        np.save(idxsfile, idxs, **args)
        return preds, targs, idxs

    # subset to boot sample from full train
    y = ds.y_train()
    boot_preds, boot_targs = [], []
    for pred, targ, idx in zip(preds, targs, ensemble_idxs):
        boot_pred, boot_targ = get_boot_samples(y, pred, targ, idx)
        boot_preds.append(boot_pred)
        boot_targs.append(boot_targ)

    preds = np.stack(boot_preds)
    targs = np.stack(boot_targs)
    idxs = np.stack(ensemble_idxs)
    np.save(predsfile, preds, **args)
    np.save(targsfile, targs, **args)
    np.save(idxsfile, idxs, **args)
    return preds, targs, idxs


if __name__ == "__main__":
    for ds in [VisionDataset.CIFAR10, VisionDataset.CIFAR100, VisionDataset.FashionMNIST]:
        for phase in FinalEvalPhase:
            preds, targs, idxs = consolidate_preds(VisionDataset.CIFAR10, phase=phase)
