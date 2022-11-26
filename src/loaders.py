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
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from src.augments import Augmenter
from src.config import Config
from src.constants import VAL_SIZE
from src.enumerables import Phase, VisionBinaryDataset, VisionDataset


class NormedDataset(TensorDataset):
    def __init__(
        self,
        *tensors: Tensor,
        train_means: ndarray,
        train_sds: ndarray,
        augment: bool = False,
    ) -> None:
        super().__init__(*tensors)
        self.augment = augment
        self.means = train_means
        self.sds = train_sds
        if augment:
            self.augmenter = Augmenter()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x, y = super().__getitem__(index)
        if self.augment:
            x = self.augmenter(x)
        # This follows the order of TA paper
        x = (x - self.means) / self.sds
        return x, y


def vision_datasets(config: Config) -> tuple[Dataset, Dataset, Dataset]:
    kind = config.vision_dataset
    binary = config.binary
    subset = config.subset  # TODO

    X = kind.x_train()
    y = kind.y_train()
    X_test = kind.x_train()
    y_test = kind.y_train()

    if X.ndim == 3:
        X = np.stack([X, X, X], axis=1)
        X_test = np.stack([X_test, X_test, X_test], axis=1)
    else:
        X = X.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)
    train_means = np.mean(X, axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)
    train_sds = np.std(X, axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)

    if binary:
        l1, l2 = kind.binary().classes()  # binary labels
        idx = (y == l1) | (y == l2)
        X, y = np.copy(X[idx]), np.copy(y[idx])  # copy defragments
        X_test, y_test = np.copy(X_test[idx]), np.copy(y_test[idx])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=69, shuffle=False
    )
    norm_args = dict(train_means=train_means, train_sds=train_sds)
    return (
        NormedDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr), augment=True, **norm_args
        ),
        NormedDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), **norm_args),
        NormedDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), **norm_args),
    )


def vision_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, val, test = vision_datasets(config)
    shared_args: Mapping = dict(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    train_loader = DataLoader(train, shuffle=True, **shared_args)
    val_loader = DataLoader(val, shuffle=False, **shared_args)
    test_loader = DataLoader(test, shuffle=False, **shared_args)  # type: ignore
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    for ds in VisionDataset:
        config = Config.from_args(f"--data={ds.value}")[0]
        vision_loaders(config)