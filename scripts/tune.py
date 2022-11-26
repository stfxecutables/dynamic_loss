from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os

from sklearn.model_selection import ParameterGrid

from src.train import evaluate

ID = os.environ.get("SLURM_ARRAY_TASK_ID")
if ID is None:
    print("WARNING: No $SLURM_ARRAY_TASK_ID found. Setting to 0")
    INDEX: int = 0
else:
    INDEX = int(ID)

LRS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
WDS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
GRID = list(ParameterGrid(dict(lr=LRS, wd=WDS)))  # 30
LR = GRID[INDEX]["lr"]
WD = GRID[INDEX]["wd"]

if __name__ == "__main__":
    print(f"Evaluating CIFAR-100 with LR={LR}, WD={WD}")
    print(f"Grid search index {INDEX} of {len(GRID)}")
    evaluate(f"--dataset=cifar-100 --batch_size=1024 --num_workers=1 --lr={LR} --wd={WD}")
