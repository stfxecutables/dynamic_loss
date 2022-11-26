from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


DATA = ensure_dir(ROOT / "data")
LOG_ROOT_DIR = ensure_dir(ROOT / "logs")

VAL_SIZE = 0.2
"""
CIFAR-10
    At batch=256, WR16-10 w/ default lr, wd hits  0.739 val acc at 10 epochs
    At batch=256, WR16-10 w/ default lr, wd hits  0.884 val acc at 21 epochs
    At batch=256, WR16-10 w/ default lr, wd hits  0.892 val acc at 30 epochs
    30 epochs = ~25 minutes

"""
# BATCH_SIZE =

"""
[General]
Dataset = MNIST
ValidationProportion = 0.2


[Ensemble]
Type = stacked
NumBaseLearners = 3
UseDynamicLoss = True
DynamicLossThreshold = 0.7


[GA]
GAGenerations = 300
GAPopulation = 200

"""