from __future__ import annotations

# fmt: off
import sys  # isort:skip
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
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import nll_loss, relu
from typing_extensions import Literal


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
    threshold: float | None, soften: bool, r: float = 0.1
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
        # preds = preds.squeeze()
        if preds.ndim != 2:
            raise ValueError(f"Invalid preds shape: {preds.shape}.")
        logits = torch.softmax(preds, dim=1)
        logits.retain_grad()
        ones = torch.ones_like(logits, requires_grad=True)
        shrunk = r * logits
        if soften:
            scaled = torch.where(logits > threshold, torch.pow(logits, 0.1), shrunk)
        else:
            scaled = torch.where(logits > threshold, ones, shrunk)
        loss_ = nll_loss(torch.log(scaled), target)
        loss_.retain_grad()
        return loss_, logits

    return loss


def compute_gradients(
    seed: int,
    thresh: float,
    r: float = 0.1,
) -> tuple[
    Tensor, Tensor, ndarray, ndarray, Tensor, Tensor, Tensor, Linear, Linear, Linear
]:
    torch.manual_seed(seed)

    # updated seeds
    # torch.manual_seed(16)  # prediction is correct, and above threshold
    dyn_loss = dynamic_loss(thresh, soften=False, r=r)
    dyn_soft_loss = dynamic_loss(thresh, soften=True, r=r)

    # out final "classifier" linear layer
    lin_args: Mapping = dict(in_features=4, out_features=3, bias=False)
    linear = Linear(**lin_args)
    w = linear.weight
    w.retain_grad()

    # make copies with identical weights init to see gradients from diff losses
    w1 = w.clone().detach()
    w2 = w.clone().detach()
    w3 = w.clone().detach()
    w4 = w.clone().detach()
    lin1 = Linear(**lin_args)
    lin2 = Linear(**lin_args)
    lin3 = Linear(**lin_args)
    lin4 = Linear(**lin_args)
    lin1.weight = torch.nn.parameter.Parameter(w1, requires_grad=True)
    lin2.weight = torch.nn.parameter.Parameter(w2, requires_grad=True)
    lin3.weight = torch.nn.parameter.Parameter(w3, requires_grad=True)
    lin4.weight = torch.nn.parameter.Parameter(w4, requires_grad=True)
    lin1.weight.retain_grad()
    lin2.weight.retain_grad()
    lin3.weight.retain_grad()
    lin4.weight.retain_grad()

    # random input
    x = 3 * torch.randn([1, 4], requires_grad=False)

    # pass through "model"
    preds1 = lin1(x.clone())
    preds2 = lin2(x.clone())
    preds3 = lin3(x.clone())
    preds4 = lin4(x.clone())

    preds1.retain_grad()
    preds2.retain_grad()
    preds3.retain_grad()
    preds4.retain_grad()

    target = torch.randint(0, 2, size=(1,))
    dyn, softmax_dyn = dyn_loss(preds1, target)
    dyn_soft, softmax_dyn_soft = dyn_soft_loss(preds3, target)
    softmax_ce = torch.softmax(preds4, dim=1)
    ce = nll_loss(torch.log(softmax_ce), target)

    softmax_dyn.retain_grad()
    softmax_ce.retain_grad()
    softmax_dyn_soft.retain_grad()
    dyn.retain_grad()
    dyn_soft.retain_grad()
    ce.retain_grad()

    dyn.backward()
    dyn_soft.backward()
    ce.backward()

    correct = target.reshape(-1, 1).numpy()
    pred = np.argmax(softmax_dyn.detach().numpy(), axis=1).reshape(-1, 1)

    return (
        w,
        target,
        correct,
        pred,
        softmax_dyn,
        softmax_ce,
        softmax_dyn_soft,
        lin1,
        lin3,
        lin4,
    )
    x.retain_grad()


def print_gradients(
    thresh: float,
    correct: ndarray,
    pred: ndarray,
    softmax_dyn: Tensor,
    softmax_ce: Tensor,
    softmax_dyn_soft: Tensor,
    lin1: Linear,
    lin3: Linear,
    lin4: Linear,
) -> None:
    # print("Linear layer init weights:")
    # print(torch.round(w.detach(), decimals=3).numpy())
    print(f"\nSoftmaxed inputs to loss function from linear layer")
    print(torch.round(softmax_dyn.detach(), decimals=3).numpy())
    print(f"\nCorrect target:", correct.item(), "\nPrediction:    ", pred.item())
    print(f"Threshold:     {thresh}")
    # print(correct)
    # print(f"\nPrediction:")
    # print(pred)

    # print("\nGradients on raw linear outputs:")
    # print("Cross-entropy loss")
    # print(torch.round(x4.grad, decimals=5).numpy())
    # print("Dynamic loss")
    # print(torch.round(x1.grad, decimals=5).numpy())
    # print("Soft dynamic loss")
    # print(torch.round(x3.grad, decimals=5).numpy())
    # print("Dynamic loss with learnable thresholds")
    # print(torch.round(x2.grad, decimals=5).numpy())

    print("\nGradients on softmaxed variable:")
    print("Cross-entropy loss", torch.round(softmax_ce.grad, decimals=5).numpy())
    print("Dynamic loss      ", torch.round(softmax_dyn.grad, decimals=5).numpy())
    print("Soft dynamic loss ", torch.round(softmax_dyn_soft.grad, decimals=5).numpy())
    # print("Dynamic loss with learnable thresholds")
    # print(torch.round(softmax_alt.grad, decimals=4).numpy())

    print("\nGradients on linear layer weights:")
    print("Cross-entropy loss")
    print(torch.round(lin4.weight.grad, decimals=5).numpy().round(3))
    print("Dynamic loss")
    print(torch.round(lin1.weight.grad, decimals=5).numpy().round(3))
    print("Soft dynamic loss")
    print(torch.round(lin3.weight.grad, decimals=6).numpy().round(4))


def demonstrate_grad_flow(
    correct_pred: bool, confident_pred: float, r: float = 0.1
) -> None:
    """
    correct_pred: prediction == true value
    confident_pred: largest softmax is above threshold
    """
    thresh = 0.7

    # torch.manual_seed(7)  # to demo problem when all preds incorrect
    # torch.manual_seed(30)  # to demo correct prediction
    # torch.manual_seed(2)  # to demo problem when one pred correct, one pred incorrect

    # seeds = list(range(0, 200))
    seeds = list(range(17, 200))

    for seed in seeds:
        (
            w,
            target,
            correct,
            pred,
            softmax_dyn,
            softmax_ce,
            softmax_dyn_soft,
            lin1,
            lin3,
            lin4,
        ) = compute_gradients(seed, thresh=thresh, r=r)

        # make align with `correct_pred` case
        is_correct = bool(correct.item() == pred.item())
        if correct_pred and (not is_correct):
            continue

        is_confident = max(*softmax_dyn.detach().ravel().tolist()) > thresh
        if not (is_confident is confident_pred):
            continue

        print("\n")
        print("=" * 80)
        if correct_pred and confident_pred:
            print(f"CORRECT, CONFIDENT (above threshold) prediction (seed={seed}, r={r})")
        elif correct_pred and (not confident_pred):
            print(f"CORRECT, DOUBTFUL (below threshold) prediction (seed={seed}, r={r})")
        elif (not correct_pred) and confident_pred:
            print(
                f"INCORRECT, OVERCONFIDENT (above threshold) prediction (seed={seed}, r={r})"
            )
        elif (not correct_pred) and (not confident_pred):
            print(
                f"INCORRECT, DOUBTFUL (below threshold) prediction (seed={seed}, r={r})"
            )
        else:
            raise RuntimeError("???")
        print("=" * 80)

        print_gradients(
            thresh=thresh,
            correct=correct,
            pred=pred,
            softmax_dyn=softmax_dyn,
            softmax_ce=softmax_ce,
            softmax_dyn_soft=softmax_dyn_soft,
            lin1=lin1,
            lin3=lin3,
            lin4=lin4,
        )

        break


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
    demonstrate_grad_flow(correct_pred=True, confident_pred=True)
    demonstrate_grad_flow(correct_pred=True, confident_pred=True, r=0.5)

    demonstrate_grad_flow(correct_pred=True, confident_pred=False)
    demonstrate_grad_flow(correct_pred=True, confident_pred=False, r=0.5)

    demonstrate_grad_flow(correct_pred=False, confident_pred=True)
    demonstrate_grad_flow(correct_pred=False, confident_pred=True, r=0.5)

    demonstrate_grad_flow(correct_pred=False, confident_pred=False)
    demonstrate_grad_flow(correct_pred=False, confident_pred=False, r=0.5)

    # plot_losses()