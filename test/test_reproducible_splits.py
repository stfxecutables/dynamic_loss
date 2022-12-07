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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.constants import N_ENSEMBLES
from src.enumerables import VisionDataset
from src.loaders.loaders import get_train_val_splits, to_BCHW

OUTDIR = Path(__file__).resolve().parent


def create_splits(kind: VisionDataset, ensemble_idx: int) -> tuple[ndarray, ndarray]:
    X = kind.x_train()
    y = kind.y_train()
    X = to_BCHW(X)
    _, _, y_tr, y_val = get_train_val_splits(X, y, ensemble_idx=ensemble_idx)
    y_tr_out = OUTDIR / f"y_tr_{kind.name}_{ensemble_idx}.npy"
    if not y_tr_out.exists():
        np.save(OUTDIR / f"y_tr_{kind.name}_{ensemble_idx}.npy", y_tr)
        np.save(OUTDIR / f"y_val_{kind.name}_{ensemble_idx}.npy", y_val)
    return y_tr, y_val


def load_splits(kind: VisionDataset, ensemble_idx: int) -> tuple[ndarray, ndarray]:
    y_tr_out = OUTDIR / f"y_tr_{kind.name}_{ensemble_idx}.npy"
    if y_tr_out.exists():
        y_tr = np.load(OUTDIR / f"y_tr_{kind.name}_{ensemble_idx}.npy")
        y_val = np.load(OUTDIR / f"y_val_{kind.name}_{ensemble_idx}.npy")
    else:
        raise FileNotFoundError()
    return y_tr, y_val


def test_reproducible_splits() -> None:
    for kind in VisionDataset:
        for idx in range(N_ENSEMBLES):
            y_tr, y_val = create_splits(kind, idx)
            y_tr_, y_val_ = load_splits(kind, idx)
            np.testing.assert_array_equal(y_tr, y_tr_)
            np.testing.assert_array_equal(y_val, y_val_)


if __name__ == "__main__":
    # run twice
    test_reproducible_splits()
