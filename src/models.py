from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
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
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from numpy import ndarray
from pandas import DataFrame, Series
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.autograd import Variable
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    Identity,
    LeakyReLU,
    Linear,
    Module,
    Sequential,
)
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy
from torchvision.models.resnet import Bottleneck, ResNet, _ovewrite_named_param, _resnet
from typing_extensions import Literal

from src.config import Config
from src.dynamic_loss import DynamicThresholder, dynamic_loss, thresholded_loss
from src.enumerables import FinalEvalPhase, Loss, Phase
from src.metrics import Metrics
from src.wideresnet import WideResNet


class BaseModel(LightningModule):
    def __init__(
        self,
        config: Config,
        log_version_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model: Module
        self.config = config
        self.num_classes = config.num_classes
        self.train_metrics = Metrics(self.config, Phase.Train)
        self.val_metrics = Metrics(self.config, Phase.Val)
        self.test_metrics = Metrics(self.config, Phase.Test)
        self.thresholder = Identity()

        self.ce_loss = CrossEntropyLoss()
        if self.config.loss is Loss.CrossEntropy:
            self.loss = CrossEntropyLoss()
        elif self.config.loss is Loss.DynamicLoss:
            self.loss = dynamic_loss(
                self.config.loss_threshold, soften=self.config.soften
            )
        elif self.config.loss is Loss.DynamicTrainable:
            self.thresholder = DynamicThresholder()
            self.loss = thresholded_loss
        elif self.config.loss is Loss.DynamicFirstTrainable:
            self.thresholder = DynamicThresholder(
                trainable_threshold=True,
                trainable_scale_factor=False,
            )
            self.loss = CrossEntropyLoss()
        elif self.config.loss is Loss.DynamicFirst:
            thresh = self.config.loss_threshold
            self.thresholder = DynamicThresholder(
                threshold=thresh,
                trainable_threshold=False,
                trainable_scale_factor=False,
            )
            self.loss = CrossEntropyLoss()
        else:
            raise ValueError()
        self.log_version_dir: Path = log_version_dir
        self.final_eval: FinalEvalPhase | None = None

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        if self.config.loss in [Loss.DynamicFirst, Loss.DynamicFirstTrainable]:
            x = self.thresholder(x)
            x = self.model(x)
        else:
            x = self.model(x)
            x = self.thresholder(x)
        return x

    @no_type_check
    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Tensor:
        preds, loss = self._shared_step(batch)[:2]
        if batch_idx % 20 == 0 and batch_idx != 0:
            self.train_metrics.log(self, preds=preds, target=batch[1])
        self.log(f"{Phase.Train.value}/loss", loss, on_step=True)
        return loss  # auto-logged by Lightning

    @no_type_check
    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Any:
        preds, loss, target = self._shared_step(batch)
        if self.final_eval is not None:
            return {
                "pred": preds.cpu().numpy(),
                "target": batch[1].cpu().numpy(),
            }
        self.val_metrics.log(self, preds, target)
        self.log(f"{Phase.Val.value}/loss", loss, prog_bar=True)

    @no_type_check
    def validation_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:
        """Save predictions each epoch. We will compare to true values after."""
        if self.trainer is None:
            raise RuntimeError(f"LightningModule {self} has empty .trainer property")
        if self.final_eval is None:
            return
        preds = [output["pred"] for output in outputs]
        targs = [output["target"] for output in outputs]

        preds = np.concatenate(preds, axis=0)
        targs = np.concatenate(targs, axis=0)
        acc = accuracy(
            torch.from_numpy(preds), torch.from_numpy(targs), num_classes=self.num_classes
        )

        epoch = int(self.current_epoch)
        logdir = self.log_version_dir
        phase = self.final_eval.value
        outdir = logdir / f"{phase}_preds"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"{phase}_preds_epoch={epoch:03d}.npy"
        np.save(outfile, preds)
        outfile = outdir / f"{phase}_labels_epoch={epoch:03d}.npy"
        np.save(outfile, targs)
        outfile = outdir / f"{phase}_acc={acc:0.4f}_epoch={epoch:03d}.npy"
        np.save(outfile, acc)

    @no_type_check
    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Tensor:
        preds, loss = self._shared_step(batch)[:2]
        self.test_metrics.log(self, preds=preds, target=batch[1])
        self.log(f"{Phase.Test.value}/loss", loss)
        return {
            "pred": preds.cpu().numpy(),
            "loss": loss.cpu().numpy(),
            "target": batch[1].cpu().numpy(),
        }

    @no_type_check
    def test_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:
        """Save predictions each epoch. We will compare to true values after."""
        if self.trainer is None:
            raise RuntimeError(f"LightningModule {self} has empty .trainer property")
        preds = [output["pred"] for output in outputs]
        targs = [output["target"] for output in outputs]
        losses = [output["loss"] for output in outputs]

        preds = np.concatenate(preds, axis=0)
        targs = np.concatenate(targs, axis=0)
        losses = np.ravel(losses)
        acc = accuracy(
            torch.from_numpy(preds), torch.from_numpy(targs), num_classes=self.num_classes
        )

        epoch = int(self.current_epoch)
        logdir = self.log_version_dir
        outdir = logdir / "test_preds"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"test_preds_epoch={epoch:03d}.npy"
        np.save(outfile, preds)
        outfile = outdir / f"test_labels_epoch={epoch:03d}.npy"
        np.save(outfile, targs)
        outfile = outdir / f"test_acc={acc:0.4f}_epoch={epoch:03d}.npy"
        np.save(outfile, acc)

    def _shared_step(self, batch: Tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x, target = batch
        preds = self(x)  # need pred.shape == (B, n_classes, H, W)
        if self.current_epoch < self.config.dyn_epoch:
            loss = self.ce_loss(preds, target)
        else:
            loss = self.loss(preds, target)
        return preds, loss, target

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        warmup = 2 if self.config.max_epochs <= 10 else 5
        opt = AdamW(
            self.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay,
        )
        sched = LinearWarmupCosineAnnealingLR(
            optimizer=opt,
            warmup_epochs=warmup,
            warmup_start_lr=1e-5,
            max_epochs=self.config.max_epochs,
            eta_min=1e-9,
        )
        return [opt], [sched]


class WideResNet16_8(BaseModel):
    def __init__(
        self,
        config: Config,
        log_version_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, log_version_dir=log_version_dir, *args, **kwargs)
        self.depth = 16
        self.k = 8
        self.model = WideResNet(depth=self.depth, k=self.k, num_classes=self.num_classes)


class WideResNet28_10(BaseModel):
    def __init__(
        self,
        config: Config,
        log_version_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, log_version_dir=log_version_dir, *args, **kwargs)
        self.depth = 28
        self.k = 10
        self.model = WideResNet(depth=self.depth, k=self.k, num_classes=self.num_classes)


class MlpBlock(Module):
    """Simple blocks of https://arxiv.org/pdf/1705.03098.pdf"""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = Sequential(
            Linear(width, width, bias=False),
            BatchNorm1d(width),
            LeakyReLU(),
            Dropout(0.5),
            Linear(width, width, bias=False),
            BatchNorm1d(width),
            LeakyReLU(),
            Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return x + out


class BetterLinear(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = Sequential(
            Linear(in_channels, out_channels, bias=False),
            BatchNorm1d(out_channels),
            LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MLP(BaseModel):
    """Similar to approach of https://arxiv.org/pdf/1705.03098.pdf"""

    def __init__(
        self,
        config: Config,
        log_version_dir: Path,
        in_channels: int,
        width1: int = 512,
        width2: int = 256,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, log_version_dir=log_version_dir, *args, **kwargs)
        self.model = Sequential(
            BetterLinear(in_channels=in_channels, out_channels=width1),
            MlpBlock(width=width1),
            BetterLinear(in_channels=width1, out_channels=width2),
            MlpBlock(width=width2),
            Linear(width2, self.num_classes, bias=True),
        )


class WeightedAggregator(BaseModel):
    def __init__(
        self,
        config: Config,
        log_version_dir: Path,
        in_channels: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, log_version_dir, *args, **kwargs)
        self.in_channels = in_channels
        self.model = Linear(
            in_features=in_channels, out_features=self.num_classes, bias=True
        )


if __name__ == "__main__":
    wr = wide_resnet16_10()
    print(wr)