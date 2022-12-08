from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import Resize
from typing_extensions import Literal

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
from src.models import MLP, BaseModel

LOG_DIR = LOG_ROOT_DIR if ON_CCANADA else CC_LOGS / "logs"
BOOT_DIR = LOG_DIR / "base-train/boot"


def summarize_classic_ensemble_test_accs(dataset: VisionDataset) -> DataFrame:
    preds, targs, idxs = consolidate_preds(dataset, FinalEvalPhase.Test)
    targ = targs[0]
    votes = np.argmax(preds, axis=-1)  # votes[:, i] are labels for sample i
    vote_preds = mode(votes, axis=0)[0].squeeze()
    agg_logits = np.mean(preds, axis=0)  # shape is (n_samples, n_classes)
    agg_preds = np.argmax(agg_logits, axis=1)

    all_accs = np.mean(votes == targs, axis=1)  # (n_ensembles,)
    sd_acc = np.std(all_accs, ddof=1)
    acc_min, acc_max = all_accs.min(), all_accs.max()
    acc_avg = np.mean(all_accs)

    vote_acc = np.mean(vote_preds == targ)
    agg_acc = np.mean(agg_preds == targ)
    return DataFrame(
        {
            "data": dataset.value,
            "base_avg": acc_avg,
            "base_sd": sd_acc,
            "base_min": acc_min,
            "base_max": acc_max,
            "vote": vote_acc,
            "agg": agg_acc,
        },
        index=[dataset.value],
    )


if __name__ == "__main__":
    dfs = []
    for ds in [VisionDataset.FashionMNIST, VisionDataset.CIFAR10, VisionDataset.CIFAR100]:
        dfs.append(summarize_classic_ensemble_test_accs(ds))
    df = pd.concat(dfs, axis=0)
    print(df.round(4).to_markdown(tablefmt="simple", index=False))
