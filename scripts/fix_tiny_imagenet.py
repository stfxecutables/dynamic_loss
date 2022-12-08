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
from shutil import copyfile, move, rmtree
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
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import DATA
from src.enumerables import VisionBinaryDataset, VisionDataset

TINY = DATA / "tiny_imagenet"
TINY_ROOT = TINY / "tiny-imagenet-200"
TRAIN_START = TINY_ROOT / "train"
VAL_START = TINY_ROOT / "val"
TEST_START = TINY_ROOT / "test"
STARTS = [TRAIN_START, VAL_START, TEST_START]

TRAIN = TINY / "train"
VAL = TINY / "val"
TEST = TINY / "test"  # Actually useless, no labels
ENDS = [TRAIN, VAL, TEST]

VAL_ANNOTATIONS = VAL / "val_annotations.txt"
VAL_IMAGES = VAL / "images"


def move_files() -> None:
    if TINY_ROOT.exists():
        for start_dir, end_dir in zip(STARTS, ENDS):
            if start_dir.exists():
                move(start_dir, end_dir)
        text_files = sorted(TINY_ROOT.glob("*.txt"))
        for text in text_files:
            move(text, TINY)
        try:
            TINY_ROOT.rmdir()
        except Exception as e:
            raise FileExistsError(
                f"Could not delete non-empty directory: {TINY_ROOT}"
            ) from e


def organize_val_images() -> None:
    with open(VAL_ANNOTATIONS, "r") as handle:
        lines = handle.readlines()

    clean = [line.replace("\n", "").split("\t")[:2] for line in lines]
    imgs, labels = zip(*clean)
    unq_labels = np.unique(labels).tolist()
    for unq in unq_labels:
        (VAL_IMAGES / unq).mkdir(exist_ok=True, parents=True)

    for img, label in tqdm(
        zip(imgs, labels), total=len(imgs), desc="Organizing val images"
    ):
        img_start = VAL_IMAGES / img
        img_end = VAL_IMAGES / f"{label}/{img}"
        copyfile(img_start, img_end)


if __name__ == "__main__":
    move_files()
    organize_val_images()