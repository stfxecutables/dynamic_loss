from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from typing import Any, Callable, Mapping, Type, Union, cast, no_type_check

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import Accuracy
from typing_extensions import Literal

from src.config import Config
from src.enumerables import Phase


class Metrics:
    def __init__(self, config: Config, phase: Phase) -> None:
        self.num_classes = config.num_classes
        self.phase = phase
        self.metrics = {
            "acc": Accuracy(num_classes=self.num_classes),
            "top3": Accuracy(num_classes=self.num_classes, top_k=3),
            "top5": Accuracy(num_classes=self.num_classes, top_k=5),
        }

    def log(
        self,
        lightning_module: LightningModule,
        preds: Tensor,
        target: Tensor,
    ) -> None:
        with torch.no_grad():
            preds = preds.cpu()
            target = target.cpu()
            for name, metric in self.metrics.items():
                value = metric(preds, target)
                metricname = f"{self.phase.value}/{name}"
                lightning_module.log(
                    metricname,
                    value,
                    on_step=name == "acc",
                    on_epoch=True,
                    prog_bar=name == "acc",
                )