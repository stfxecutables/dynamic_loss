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
from torch import Tensor
from typing_extensions import Literal

from src.constants import DATA


class ArgEnum(Enum):
    @classmethod
    def choices(cls) -> str:
        info = " | ".join([str(e.value) for e in cls])
        return f"< {info} >"

    @classmethod
    def choicesN(cls) -> str:
        info = " | ".join([str(e.value) for e in cls])
        return f"< {info} | None >"

    @classmethod
    def parse(cls, s: str) -> ArgEnum:
        return cls(s.lower())

    @classmethod
    def parseN(cls, s: str) -> ArgEnum | None:
        if s.lower() in ["none", "", " "]:
            return None
        return cls(s.lower())

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def names(cls) -> list[str]:
        return [e.name for e in cls]


class Phase(ArgEnum):
    Train = "train"
    Val = "val"
    Pred = "pred"
    Test = "test"


class FinalEvalPhase(ArgEnum):
    FullTrain = "train_full"
    BootTrain = "train_boot"
    Val = "val"
    Test = "test"


class VisionDataset(ArgEnum):
    MNIST = "mnist"
    FashionMNIST = "fmnist"
    CIFAR10 = "cifar-10"
    CIFAR100 = "cifar-100"
    TinyImageNet = "tiny-imagenet"

    def x_train_path(self) -> Path:
        return DATA / f"{self.value}_x_train.npy"

    def x_test_path(self) -> Path:
        return DATA / f"{self.value}_x_test.npy"

    def y_train_path(self) -> Path:
        return DATA / f"{self.value}_y_train.npy"

    def y_test_path(self) -> Path:
        return DATA / f"{self.value}_y_test.npy"

    def x_train(self) -> ndarray:
        return np.load(DATA / f"{self.value}_x_train.npy")

    def x_test(self) -> ndarray:
        return np.load(DATA / f"{self.value}_x_test.npy")

    def y_train(self) -> ndarray:
        return np.load(DATA / f"{self.value}_y_train.npy")

    def y_test(self) -> ndarray:
        return np.load(DATA / f"{self.value}_y_test.npy")

    def binary(self) -> VisionBinaryDataset:
        if self is VisionDataset.CIFAR100:
            raise ValueError("No binary dataset for CIFAR-100")
        return {
            VisionDataset.MNIST: VisionBinaryDataset.MNIST,
            VisionDataset.FashionMNIST: VisionBinaryDataset.FashionMNIST,
            VisionDataset.CIFAR10: VisionBinaryDataset.CIFAR10,
        }[self]

    def num_classes(self) -> int:
        return {
            VisionDataset.CIFAR100: 100,
            VisionDataset.MNIST: 10,
            VisionDataset.FashionMNIST: 10,
            VisionDataset.CIFAR10: 10,
        }[self]


class VisionBinaryDataset(Enum):
    MNIST = "mnist-bin"
    FashionMNIST = "fmnist-bin"
    CIFAR10 = "cifar-bin"

    def classes(self) -> list[int]:
        return {
            VisionBinaryDataset.MNIST: [4, 9],
            VisionBinaryDataset.FashionMNIST: [0, 6],  # these are right
            VisionBinaryDataset.CIFAR10: [3, 5],
        }[self]


class Experiment(ArgEnum):
    Tune = "tune"
    BaseTrain = "base-train"
    NoEnsemble = "no-ensemble"  # Baseline CNN
    BaseEnsemble = "ensemble"  # E1, E2
    DynamicLoss = "dynamic"  # E3-6
    SnapshotEnsemble = "snapshot"
    Debug = "debug"


class TrainingSubset(ArgEnum):
    """
    There is no "super-training" subset to be used for anything. The figure is
    misleading. Thus, there are only full-training sets, or the bootstrap set.
    """

    Full = "full"
    Boot = "boot"


class FusionMethod(ArgEnum):
    Vote = "vote"
    Average = "avg"  # Aggregation
    GA_Weighted = "ga-weighted"  # Genetic Algorithm weighted
    CNN = "cnn"  # "stacked" CNN
    MLP = "mlp"  # "stacked" MLP


class Loss(ArgEnum):
    CrossEntropy = "cross-entropy"
    DynamicLoss = "dynamic"