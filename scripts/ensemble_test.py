from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import traceback
from argparse import ArgumentParser, Namespace
from base64 import urlsafe_b64encode
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import rmtree
from time import sleep, strftime
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
from uuid import UUID, uuid4
from warnings import filterwarnings

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
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn import Conv1d, LeakyReLU, Linear, Module, Sequential
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from typing_extensions import Literal

from src.callbacks import ensemble_callbacks
from src.config import Config
from src.dynamic_loss import DynamicThresholder
from src.enumerables import FinalEvalPhase, FusionMethod, Loss
from src.loaders.ensemble import ensemble_loaders
from src.loaders.loaders import vision_loaders
from src.models import (
    MLP,
    BaseModel,
    WeightedAggregator,
    WideResNet,
    WideResNet16_8,
    WideResNet28_10,
)
from src.train import get_model, setup_logging


def ensemble_eval(
    argstr: str | None = None,
    threshold: float | None = None,
    pooled: bool = False,
    shuffled: bool = False,
) -> None:
    filterwarnings("ignore", message=".*does not have many workers.*")
    if argstr is None:
        config, remain = Config.from_args()
    else:
        config, remain = Config.from_args(argstr)
    logger, log_version_dir, uuid = setup_logging(config)
    train, val, test, in_channels = ensemble_loaders(
        config=config, pooled_ensembles=pooled, shuffled=shuffled, threshold=threshold
    )
    if config.fusion is FusionMethod.MLP:
        model = MLP(
            config=config, log_version_dir=log_version_dir, in_channels=in_channels
        )
    elif config.fusion is FusionMethod.Weighted:
        model = WeightedAggregator(
            config=config, log_version_dir=log_version_dir, in_channels=in_channels
        )
    parser = ArgumentParser()
    Trainer.add_argparse_args(parser)
    trainer: Trainer = Trainer.from_argparse_args(
        parser.parse_args(remain),
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        log_every_n_steps=10,
        callbacks=ensemble_callbacks(log_version_dir),
    )
    try:
        trainer.fit(model, train, val)
    except Exception as e:
        info = traceback.format_exc()
        print(info)
        print(f"Got error: {e}")
        print("Removing leftover logs...")
        rmtree(log_version_dir)
        record = log_version_dir.parent / "errors.txt"
        with open(record, "w") as handle:
            handle.write(info)
        print(
            f"Removed leftover logs. Error info in {record} Exiting with error code 1..."
        )
        sys.exit(1)
    if config.loss in [
        Loss.DynamicTrainable,
        Loss.DynamicFirst,
        Loss.DynamicFirstTrainable,
    ]:
        layer: DynamicThresholder = model.thresholder
        threshold = layer.T
        scaling = layer.r
        print(f"DynamicThresholder learned threshold: {threshold}")
        print(f"DynamicThresholder learned scale-factor: {scaling}")
    sleep(2)
    ckpt = log_version_dir / "ckpts/last.ckpt"
    trainer.test(model, test, ckpt_path="best")


if __name__ == "__main__":
    ensemble_eval(threshold=0.7)
    # we are getting 0.9775 test acc with
    # python scripts/ensemble_test.py --experiment=debug --subset=full --dataset=cifar-100 --max_epochs=100 --batch_size=1024 --lr=3e-3
    # HOLY FUCK
    # CIFAR-100: 0.9891
    # python scripts/ensemble_test.py --experiment=debug --subset=full --dataset=cifar-100 --max_epochs=75 --batch_size=1024 --lr=3e-3

    # CIFAR-10: 0.99088
    # python scripts/ensemble_test.py --experiment=debug --subset=full --dataset=cifar-10 --max_epochs=75 --batch_size=1024 --lr=3e-3
    # CIFAR-10: 0.9958
    # python scripts/ensemble_test.py --experiment=debug --subset=full --dataset=cifar-10 --max_epochs=100 --batch_size=1024 --lr=3e-3

    # FMNIST: 0.9798
    # python scripts/ensemble_test.py --experiment=debug --subset=full --dataset=fmnist --max_epochs=100 --batch_size=1024 --lr=3e-3