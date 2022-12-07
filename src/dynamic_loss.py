from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on
from typing import Callable

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.nn import Conv1d, LeakyReLU, Linear, Module, Sequential
from torch.nn.functional import leaky_relu, nll_loss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy


def dynamic_loss(threshold: float | None) -> Callable[[Tensor, Tensor], Tensor]:
    """Expects raw linear outputs (e.g. no softmax or sigmoid has
    been applied)

    Parameters
    ----------
    preds: Tensor
        Outputs from final linear or Conv1d layer, i.e. must have shape
        (B, C) or be such that `preds.squeeze` has shape (B, C), where C
        is the number of classes. Values must NOT be softmaxed or sigmoided.

    target: Tensor
        Tensor in the same form as expected for the target of
        `torch.nn.NLLLoss` (https://pytorch.org/docs/stable/generated/
        torch.nn.NLLLoss.html#torch.nn.NLLLoss).

    threshold: float
        Threshold in (0, 1].
    """
    if threshold is None:
        raise ValueError("Must have float threshold")
    if threshold <= 0:
        raise ValueError(f"Threshold must be in (0, 1]. Got {threshold}")

    def loss(preds: Tensor, target: Tensor) -> Tensor:
        preds = preds.squeeze()
        if preds.ndim != 2:
            raise ValueError(f"Invalid preds shape: {preds.shape}.")
        logits = torch.softmax(preds, dim=1)
        ones = torch.ones_like(logits, requires_grad=True)
        shrunk = 0.1 * logits
        scaled = torch.where(logits > threshold, ones, shrunk)
        return nll_loss(torch.log(scaled), target)

    return loss
