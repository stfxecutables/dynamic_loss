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
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Conv2d, LeakyReLU, Linear, Module, Sequential
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy
from torchvision.models.resnet import Bottleneck, ResNet, _ovewrite_named_param, _resnet
from typing_extensions import Literal


def wide_resnet28_10(**kwargs: Any) -> ResNet:
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [2, 2, 2, 2], weights=None, progress=False, **kwargs)


def wide_resnet16_10(**kwargs: Any) -> ResNet:
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [1, 1, 1, 1], weights=None, progress=False, **kwargs)


class WideResNet(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = wide_resnet16_10()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class WideResNet16_10(WideResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = wide_resnet16_10()


class WideResNet28_10(WideResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = wide_resnet28_10()


if __name__ == "__main__":
    wr = wide_resnet16_10()
    print(wr)