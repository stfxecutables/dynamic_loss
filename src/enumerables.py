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
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal


class Phase(Enum):
    Train = "train"
    Val = "val"
    Pred = "pred"
    Test = "test"


class VisionDataset(Enum):
    MNIST = "mnist"
    FashionMNIST = "fmnist"
    CIFAR10 = "cifar-10"
    CIFAR100 = "cifar-100"


class VisionBinaryDataset(Enum):
    MNIST = "mnist-bin"
    FashionMNIST = "fmnist-bin"
    CIFAR10 = "cifar-bin"

    def classes(self) -> list[int]:
        return {
            VisionBinaryDataset.MNIST: [4, 9],
            VisionBinaryDataset.FashionMNIST: [0, 6],
            VisionBinaryDataset.CIFAR10: [3, 5],
        }[self]