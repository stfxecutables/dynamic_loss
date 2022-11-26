from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from src.augments import Augmenter
from src.config import Config
from src.enumerables import Phase, VisionBinaryDataset, VisionDataset


class AugmentDataset(TensorDataset):
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__(*tensors)
        self.augmenter = Augmenter()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        X, y = super().__getitem__(index)
        return self.augmenter(X), y



def vision_dataset(config: Config, phase: Phase) -> Dataset:
    kind = config.vision_dataset
    binary = config.binary
    X = kind.x_train() if phase is Phase.Train else kind.x_test()
    y = kind.y_train() if phase is Phase.Train else kind.y_test()
    if X.ndim == 3:
        X = X[:, None, :, :]
    if not binary:
        return AugmentDataset(torch.from_numpy(X), torch.from_numpy(y))
    l1, l2 = kind.binary().classes()  # binary labels
    idx = (y == l1) | (y == l2)
    X, y = np.copy(X[idx]), np.copy(y[idx])  # defragment
    return AugmentDataset(torch.from_numpy(X), torch.from_numpy(y))


def vision_loaders(
    config: Config,
):
    pass


if __name__ == "__main__":
    pass