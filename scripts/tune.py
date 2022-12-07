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

if __name__ == "__main__":
    LRS = [1e-3, 1e-2, 5e-2, 1e-1]
    WDS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    GRID = list(ParameterGrid(dict(lr=LRS, wd=WDS, resize=[128])))  # 28
    print(f"Total number of combinations: {len(GRID)}")

    ID = os.environ.get("SLURM_ARRAY_TASK_ID")
    if ID is None:
        sys.exit()
    else:
        INDEX = int(ID)

    LR = GRID[INDEX]["lr"]
    WD = GRID[INDEX]["wd"]
    RESIZE = GRID[INDEX]["resize"]
    print(f"Evaluating CIFAR-100 with LR={LR}, WD={WD}")
    print(f"Grid search index {INDEX} of {len(GRID)}")
    evaluate(
        f"--experiment=tune --subset=full --dataset=cifar-100 --loss=cross-entropy --augment=True --batch_size=1024 --max_epochs=50 --num_workers=1 --lr={LR} --wd={WD} --resize={RESIZE}",
    )