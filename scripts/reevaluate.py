from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
from pathlib import Path

from src.constants import LOG_ROOT_DIR
from src.enumerables import Experiment
from src.train import reevaluate

LOGS = LOG_ROOT_DIR / Experiment.BaseTrain.value
CKPTS = sorted(LOGS.rglob("last.ckpt"))

if __name__ == "__main__":
    ID = os.environ.get("SLURM_ARRAY_TASK_ID")
    if ID is None:
        sys.exit(1)
    else:
        INDEX = int(ID)
    print(f"Re-evaluating checkpoint {INDEX} of {len(CKPTS)}...")
    ckpt = CKPTS[INDEX]
    reevaluate(ckpt)
