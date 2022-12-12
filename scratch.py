from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
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
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn.functional import nll_loss, relu
from typing_extensions import Literal

from src.dynamic_loss import dynamic_loss


def loss_alt(threshold: float, r: float) -> Tensor:
    """
    Notes
    -----
    Let f(x) = ((r * x) / (T - x)) * relu(T - x), for down-scaling factor r
    and threshold T. Then:

        If x < T  then  T - x > 0. Thus:

        ==>   relu(T - x)  ==  T - x > 0
        ==> ((r * x) / (T - x)) * relu(T - x)  ==  ((r * x) / (T - x)) * (T - x)
        ==>                                    ==  r * x
        ==> f(x) == r * x if x <= T

        If x > T, then T - x < 0. Thus:

        ==>   relu(T - x)  ==  0
        ==> ((r * x) / (T - x)) * relu(T - x)  ==  ((r * x) / (T - x)) * 0
        ==> f(x) == 0 if x > T

    For numerical stability, we could add:

        f(x) = ((r * x) / (T - x + eps)) * relu(T - x)



    """

    T = torch.tensor(threshold, requires_grad=True)
    r = torch.tensor(r, requires_grad=True)

    def loss(preds: Tensor, target: Tensor) -> Tensor:
        x = torch.softmax(preds, dim=1)
        scaled = ((r * x) / (T - x)) * relu(T - x) + 1e-8
        return nll_loss(torch.log(scaled), target), x

    return loss


def dynamic_loss(
    threshold: float | None, soften: bool
) -> Callable[[Tensor, Tensor], Tensor]:
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
        if soften:
            scaled = torch.where(logits > threshold, torch.pow(logits, 0.1), shrunk)
        else:
            scaled = torch.where(logits > threshold, ones, shrunk)
        return nll_loss(torch.log(scaled), target), logits

    return loss


def demonstrate_grad_flow() -> None:
    thresh = 0.3
    dyn_loss = dynamic_loss(thresh, soften=False)
    dyn_soft_loss = dynamic_loss(thresh, soften=True)
    alt_loss = loss_alt(thresh, r=0.1)

    # torch.manual_seed(5)  # to demo problem
    x = 3 * torch.randn([2, 2], requires_grad=True)
    x1 = x.clone()
    x2 = x.clone()
    x3 = x.clone()
    x4 = x.clone()

    x1.retain_grad()
    x2.retain_grad()
    x3.retain_grad()
    x4.retain_grad()

    target = torch.randint(0, 2, (2,))
    dyn, softmax_dyn = dyn_loss(x1, target)
    alt, softmax_alt = alt_loss(x2, target)
    dyn_soft, softmax_dyn_soft = dyn_soft_loss(x3, target)
    softmax_ce = torch.softmax(x4, dim=1)
    ce = nll_loss(torch.log(softmax_ce), target)

    softmax_dyn.retain_grad()
    softmax_alt.retain_grad()
    softmax_ce.retain_grad()
    softmax_dyn_soft.retain_grad()

    print(f"\nSoftmaxed inputs to loss function (threshold={thresh})")
    print(torch.round(softmax_dyn.detach(), decimals=3).numpy())
    dyn.backward(retain_graph=True)
    alt.backward(retain_graph=True)
    ce.backward(retain_graph=True)
    dyn_soft.backward(retain_graph=True)

    print(f"\nCorrect targets:")
    correct = target.reshape(-1, 1).numpy()
    print(f"\nCorrect targets:")
    print(correct)
    print(f"\nPreds after argmax:")
    pred = np.argmax(softmax_dyn.detach().numpy(), axis=1).reshape(-1, 1)
    print(pred)

    print("\nGradients on raw linear outputs:")
    print("Dynamic loss")
    print(torch.round(x1.grad.mT, decimals=5).numpy())
    print("Dynamic loss with learnable thresholds")
    print(torch.round(x2.grad.mT, decimals=5).numpy())
    print("Soft dynamic loss")
    print(torch.round(x3.grad.mT, decimals=5).numpy())
    print("Cross-entropy loss")
    print(torch.round(x4.grad.mT, decimals=5).numpy())

    print("\nGradients on softmaxed variable:")
    print("Dynamic loss")
    print(torch.round(softmax_dyn.grad, decimals=4).numpy())
    print("Dynamic loss with learnable thresholds")
    print(torch.round(softmax_alt.grad, decimals=4).numpy())
    print("Soft dynamic loss")
    print(torch.round(softmax_dyn_soft.grad, decimals=4).numpy())
    print("Cross-entropy loss")
    print(torch.round(softmax_ce.grad, decimals=4).numpy())


def plot_losses() -> None:
    thresholds = [0.25, 0.5, 0.75, 0.9]
    x = torch.linspace(0, 1, 500, dtype=torch.float)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, (thresh, ax) in enumerate(zip(thresholds, axes.flat)):
        dyn = torch.where(x > thresh, torch.ones_like(x), 0.1 * x)
        smooth = torch.where(x > thresh, torch.pow(x, 0.1), 0.1 * x)
        ax.plot(x, dyn, color="red", label="hard dynamic loss" if i == 0 else None)
        ax.plot(x, smooth, color="black", label="soft dynamic loss" if i == 0 else None)
        ax.set_title(f"threshold = {thresh}")
    fig.legend().set_visible(True)
    plt.show()


if __name__ == "__main__":
    demonstrate_grad_flow()
    # plot_losses()