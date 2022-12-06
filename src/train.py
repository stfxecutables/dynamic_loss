from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from argparse import ArgumentParser, Namespace
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import strftime
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

from src.callbacks import callbacks
from src.config import Config
from src.loaders import vision_loaders
from src.models import BaseModel, WideResNet, WideResNet16_8, WideResNet28_10


def setup_logging(
    config: Config, tune: bool = False
) -> Tuple[TensorBoardLogger, Path, UUID]:
    """Create TensorBoardLogger and ensure directories are present"""
    # see
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters # noqa
    # for correct setup of hparams tab in Tensorboard
    uuid = uuid4()
    short_uuid = uuid.hex[:8]
    date = urlsafe_b64encode(strftime("%d%H%M%S").encode()).decode("ascii").rstrip("=")
    short_uuid = urlsafe_b64encode(uuid.bytes).decode("ascii").rstrip("=")[:8]
    unq = f"{date}{short_uuid}"
    logger = TensorBoardLogger(
        save_dir=config.log_base_dir(tune=tune),
        name=unq,  # change this is if you want subfolders
        version=None,
        log_graph=False,
        default_hp_metric=False,
    )
    log_version_dir = Path(logger.log_dir).resolve()
    log_version_dir.mkdir(exist_ok=True, parents=True)
    print("Run info will be logged at:", log_version_dir)
    logger.log_hyperparams(config.loggable())
    config.to_json(log_version_dir)
    print(f"Saved config to {log_version_dir}")
    return logger, log_version_dir, uuid


def get_model(config: Config) -> BaseModel:
    return WideResNet16_8(config)


def evaluate(argstr: str | None = None, tune: bool = False) -> None:
    filterwarnings("ignore", message=".*does not have many workers.*")
    if argstr is None:
        config, remain = Config.from_args()
    else:
        config, remain = Config.from_args(argstr)
    logger, log_version_dir, uuid = setup_logging(config, tune=tune)
    train, val, test = vision_loaders(config=config)
    model: BaseModel = get_model(config)
    parser = ArgumentParser()
    Trainer.add_argparse_args(parser)
    trainer: Trainer = Trainer.from_argparse_args(
        parser.parse_args(remain),
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        log_every_n_steps=10,
        callbacks=callbacks(log_version_dir),
    )
    trainer.fit(model, train, val)
    model.final_val = True
    trainer.validate(model, val, ckpt_path="last")
    trainer.test(model, test, ckpt_path="last")


if __name__ == "__main__":
    evaluate()