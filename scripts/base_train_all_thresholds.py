from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
from argparse import Namespace

from sklearn.model_selection import ParameterGrid

from src.constants import N_ENSEMBLES, OPTIMAL_LR, OPTIMAL_WD
from src.enumerables import VisionDataset
from src.train import evaluate

if __name__ == "__main__":
    GRID = [
        Namespace(**g)
        for g in list(
            ParameterGrid(
                {
                    "idx": range(N_ENSEMBLES),
                    "data": [
                        VisionDataset.CIFAR10,
                        VisionDataset.CIFAR100,
                        VisionDataset.FashionMNIST,
                    ],
                    "threshold": [0.6, 0.8, 0.9],
                }
            )
        )
    ]  # 450
    print(f"Number of combinations: {len(GRID)}")

    ID = os.environ.get("SLURM_ARRAY_TASK_ID")
    if ID is None:
        sys.exit()
    else:
        INDEX = int(ID)

    g = GRID[INDEX]
    THRESH = g.threshold
    args = (
        "--experiment=base-train "
        "--subset=boot "
        f"--ensemble_idx={g.idx} "
        f"--dataset={g.data.value} "
        "--loss=dynamic "
        f"--loss_threshold={THRESH} "
        "--augment=True "
        "--resize=128 "
        f"--lr={OPTIMAL_LR/10} "
        f"--wd={OPTIMAL_WD} "
        f"--max_epochs=50 "
        "--batch_size=1024 "
        "--num_workers=1"
    )
    print(f"Running with command:")
    print(args)
    evaluate(argstr=args, label=f"threshold_{THRESH:0.2f}")


"""
./run_python.sh src/train.py --experiment=base-train --subset=boot --ensemble_idx=0 --dataset=fmnist --loss=cross-entropy --loss_threshold=0.7 --augment=True --max_epochs=50 --batch_size=1024 --num_workers=1
"""