from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from pathlib import Path

from src.constants import LOG_ROOT_DIR
from src.enumerables import Experiment
from src.train import reevaluate

LOGS = LOG_ROOT_DIR / Experiment.BaseTrain.value
CKPTS = sorted(LOGS.rglob("last.ckpt"))

if __name__ == "__main__":
    for ckpt in CKPTS:
        reevaluate(ckpt)
