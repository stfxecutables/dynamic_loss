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
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.init import constant_, kaiming_normal_
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy
from typing_extensions import Literal


# Adapted from
# https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/wide_resnet.py
class WideBasicBlock(Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float, stride: int = 1):
        super().__init__()
        self.bn1 = BatchNorm2d(in_ch)
        self.relu = ReLU(inplace=True)
        self.conv1 = Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.dropout = Dropout(p=dropout)
        self.bn2 = BatchNorm2d(out_ch)
        self.conv2 = Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.skip = Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = Sequential(
                Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + self.skip(x)
        return out  # type: ignore


# Adapted from
# https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/wide_resnet.py
# But with obscurantism, unnecessary/incompetent arguments and argument passing,
# and other uglinesses removed.
class WideResNet(Module):
    INIT_CH = 16

    def __init__(self, depth: int, k: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.depth = depth
        self.k = k
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_basic_blocks = int(depth - 4) // 6
        if int((depth - 4) / 6) != ((depth - 4) // 6):
            raise ValueError(
                "`depth - 4` must be evenly divisible by 6, e.g depth in [10, 16, 22, 28, 34, ...]"
            )
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=self.INIT_CH,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.in_planes = self.INIT_CH  # Changed by block constructors!!!
        self.block1 = self.make_block(out_ch=widths[1], stride_init=1)
        self.block2 = self.make_block(out_ch=widths[2], stride_init=2)
        self.block3 = self.make_block(out_ch=widths[3], stride_init=2)
        self.bn1 = BatchNorm2d(widths[3])
        self.relu = ReLU(inplace=True)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.linear = Linear(in_features=widths[3], out_features=num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def make_block(self, out_ch: int, stride_init: int) -> Sequential:
        layers = []
        for i in range(self.num_basic_blocks):
            layers.append(
                WideBasicBlock(
                    in_ch=self.in_planes,
                    out_ch=out_ch,
                    dropout=self.dropout,
                    stride=stride_init if i == 0 else 1,
                )
            )
            self.in_planes = out_ch
        return Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == "__main__":
    x = torch.randn([1, 3, 32, 32]).to(device="cuda")
    model = WideResNet(16, 8, num_classes=10).cuda()
    model(x)
    model = WideResNet(16, 10, num_classes=10).cuda()
    model(x)
    model = WideResNet(28, 10, num_classes=10).cuda()
    model(x)