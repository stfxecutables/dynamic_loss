from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import traceback
from argparse import ArgumentParser, Namespace
from base64 import urlsafe_b64encode
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
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
from src.constants import RESULTS
from src.dynamic_loss import DynamicThresholder
from src.enumerables import FinalEvalPhase, FusionMethod, Loss, VisionDataset
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


def log_results(
    val_results_last: dict[str, Any],
    test_results_last: dict[str, Any],
    val_results_best: dict[str, Any],
    test_results_best: dict[str, Any],
    best_epoch: int,
    config: Config,
    threshold: float | None,
    pooled: bool,
    shuffled: bool,
) -> None:
    outdir = RESULTS / "ensemble_evals"
    outdir = outdir / "learned"
    outdir = outdir / "each"
    outdir.mkdir(exist_ok=True, parents=True)
    finished = datetime.utcnow().strftime("%Y.%m.%d--%H-%M-%S.%f")
    outfile = outdir / f"{finished}.json"

    df = DataFrame(
        {
            "data": config.vision_dataset.name,
            "fusion": config.fusion.name,
            "pooled": pooled,
            "shuffled": shuffled,
            "thresh": threshold if threshold is not None else -1,
            "acc_test_l": test_results_last["test/acc_epoch"],
            "top3_test_l": test_results_last["test/top3"],
            "top5_test_l": test_results_last["test/top5"],
            "loss_test_l": test_results_last["test/loss"],
            "acc_val_l": val_results_last["val/acc_epoch"],
            "top3_val_l": val_results_last["val/top3"],
            "top5_val_l": val_results_last["val/top5"],
            "loss_val_l": val_results_last["val/loss"],
            "acc_test_b": test_results_best["test/acc_epoch"],
            "top3_test_b": test_results_best["test/top3"],
            "top5_test_b": test_results_best["test/top5"],
            "loss_test_b": test_results_best["test/loss"],
            "acc_val_b": val_results_best["val/acc_epoch"],
            "top3_val_b": val_results_best["val/top3"],
            "top5_val_b": val_results_best["val/top5"],
            "loss_val_b": val_results_best["val/loss"],
            "meta_epochs_l": config.max_epochs,
            "meta_epochs_b": best_epoch,
            "lr": config.lr_init,
            "wd": config.weight_decay,
        },
        index=[finished],
    )
    df.to_json(outfile)
    print(f"Saved results to {outfile}")
    return df


def ensemble_eval(
    argstr: str | None = None,
    threshold: float | None = None,
    pooled: bool = False,
    shuffled: bool = False,
) -> None:
    filterwarnings("ignore", message=".*does not have many workers.*")
    if argstr is None:
        config = Config.from_args()[0]
    else:
        config = Config.from_args(argstr)[0]
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
    trainer: Trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        callbacks=ensemble_callbacks(log_version_dir, max_epochs=config.max_epochs),
        max_epochs=config.max_epochs,
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
    # if config.loss in [
    #     Loss.DynamicTrainable,
    #     Loss.DynamicFirst,
    #     Loss.DynamicFirstTrainable,
    # ]:
    #     layer: DynamicThresholder = model.thresholder
    #     threshold = layer.T
    #     scaling = layer.r
    #     print(f"DynamicThresholder learned threshold: {threshold}")
    #     print(f"DynamicThresholder learned scale-factor: {scaling}")
    sleep(2)  # leave time for ckpt to save...
    ckpts = log_version_dir / "ckpts"
    ckpt_last = str(ckpts / "last.ckpt")
    best_path = list(ckpts.glob("epoch*.ckpt"))[0]
    ckpt_best = str(best_path)
    best_epoch = int(re.search(r"epoch=(\d+)-val", best_path.name)[1]) + 1
    # trainer.test(model, test, ckpt_path="best")
    all_val_results_last = trainer.validate(model, val, ckpt_path=ckpt_last)
    all_test_results_last = trainer.test(model, test, ckpt_path=ckpt_last)
    all_val_results_best = trainer.validate(model, val, ckpt_path=ckpt_best)
    all_test_results_best = trainer.test(model, test, ckpt_path=ckpt_best)
    if isinstance(all_val_results_last, list):
        val_results_last = all_val_results_last[0]
    if isinstance(all_test_results_last, list):
        test_results_last = all_test_results_last[0]
    if isinstance(all_val_results_best, list):
        val_results_best = all_val_results_best[0]
    if isinstance(all_test_results_best, list):
        test_results_best = all_test_results_best[0]

    log_results(
        val_results_last=val_results_last,
        test_results_last=test_results_last,
        val_results_best=val_results_best,
        test_results_best=test_results_best,
        best_epoch=best_epoch,
        config=config,
        threshold=threshold,
        pooled=pooled,
        shuffled=shuffled,
    )
    print(f"Removing checkpoints...")
    rmtree(ckpts)


def get_epochs(fusion: FusionMethod, pooled: bool) -> int:
    if fusion is FusionMethod.Weighted and pooled:
        return 5
    if fusion is FusionMethod.Weighted and not pooled:
        return 10
    if fusion is FusionMethod.MLP and pooled:
        return 10
    if fusion is FusionMethod.MLP and not pooled:
        return 20
    raise ValueError()


if __name__ == "__main__":
    for ds in [VisionDataset.CIFAR10, VisionDataset.CIFAR100, VisionDataset.FashionMNIST]:
        for fusion in [FusionMethod.MLP, FusionMethod.Weighted]:
            for threshold in [None, 0.6, 0.7, 0.8, 0.9]:
                for pooled in [True, False]:
                    for shuffled in [True, False]:
                        if pooled and shuffled:
                            continue
                        max_epochs = get_epochs(fusion, pooled)
                        argstr = (
                            "--experiment=ensemble-eval "
                            "--subset=full "
                            f"--dataset={ds.value} "
                            f"--fusion={fusion.value} "
                            f"--max_epochs={max_epochs} "
                            "--batch_size=1024 "
                            "--lr=3e-4 "
                            "--num_workers=0"
                        )
                        print(f"Evaluating with args:")
                        print(argstr, f"pooled={pooled}", f"shuffled={shuffled}")
                        ensemble_eval(
                            argstr=argstr,
                            threshold=threshold,
                            pooled=pooled,
                            shuffled=shuffled,
                        )
                        sys.exit()
