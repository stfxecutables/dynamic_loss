from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import rmtree
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.nn import Conv1d, LeakyReLU, Linear, Module, Sequential
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from typing_extensions import Literal

from src.constants import DATA
from src.enumerables import VisionBinaryDataset, VisionDataset


def unify(array_like: Any) -> ndarray:
    if isinstance(array_like, Tensor):
        return array_like.numpy()
    if isinstance(array_like, ndarray):
        return array_like
    raise ValueError(f"Unrecognized type: {type(array_like)} ({array_like})")

def download() -> None:
    helpers = {
        VisionDataset.MNIST: MNIST,
        VisionDataset.FashionMNIST: FashionMNIST,
        VisionDataset.CIFAR10: CIFAR10,
        VisionDataset.CIFAR100: CIFAR100,
    }
    for vision_dataset, helper in helpers.items():
        if not vision_dataset.y_train().exists():
            train = helper(DATA, train=True, download=True, transform=unify)
            np.save(vision_dataset.x_train(), train.data, allow_pickle=False)
            np.save(vision_dataset.y_train(), train.targets, allow_pickle=False)
        if not vision_dataset.y_test().exists():
            test = helper(DATA, train=False, download=True, transform=unify)
            np.save(vision_dataset.x_test(), test.data, allow_pickle=False)
            np.save(vision_dataset.y_test(), test.targets, allow_pickle=False)

def cleanup() -> None:
    leftovers: list[Path] = [
        DATA / "cifar-10-python.tar.gz",
        DATA / "cifar-100-python.tar.gz",
        DATA / "MNIST",
        DATA / "FashionMNIST",
        DATA / "cifar-10-batches-py",
        DATA / "cifar-100-python",
    ]
    for leftover in leftovers:
        if leftover.is_dir():
            rmtree(leftover)
        else:
            leftover.unlink()

if __name__ == "__main__":
    download()
    cleanup()
