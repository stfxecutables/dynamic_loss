from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


DATA = ensure_dir(ROOT / "data")
LOG_ROOT_DIR = ensure_dir(ROOT / "logs")
CC_LOGS = ensure_dir(ROOT / "cc_logs")
ON_CCANADA = os.environ.get("CC_CLUSTER") is not None
REPRO_DIR = ensure_dir(ROOT / "reproducibility")
RESULTS = ensure_dir(ROOT / "results")

N_ENSEMBLES: int = 50
VAL_SIZE = 0.2
"""
Batch size of 1024 is viable for WR-16-10
CIFAR-10
    At batch=256, WR16-10 w/ default lr, wd hits  0.739 val acc at 10 epochs
    At batch=256, WR16-10 w/ default lr, wd hits  0.884 val acc at 21 epochs
    At batch=256, WR16-10 w/ default lr, wd hits  0.892 val acc at 30 epochs
    30 epochs = ~25 minutes
    No point in using more than 1 worker:

        4 workers = ~45sec/epoch @1024
        3 workers = ~43sec epoch @1024
        2 workers = ~43sec epoch @1024
        1 workers = ~43sec epoch @1024
        0 workers = bad

Batch size of 1024 also shockingly viable for WR-28-10 @ 128x128, but a single
epoch now takes ~1min 45sec.

At batch of 896, on tiny ImageNet, epoch takes 3:30
"""
BATCH_SIZE = 1024
OPTIMAL_WD = 0.05
OPTIMAL_LR = 0.1
BATCH_SIZES = {  # map from resize to batch size, for v100l
    32: 1024,
    128: 1024,  # and still only about a minute per epoch
}

SHUFFLE_SEED = int.from_bytes(
    b"\xfe\xeb$\xd17~s\xd5\xea\x0ba\xc4\x9d\xb4y\xf9\xf8\x92de\xbb\x0bg\xf5\xb8\xe9\x16\x9aL\x17\x96\x9fk\x94\x14\xf9\xd1\xca\\\t\xa9N\x0f\xa8\t\xdbE\xa0\xb8\x86&\xc8\xb1\xb5\x8d9,h\xd8\x17Q5_\x7f",
    byteorder="big",
)

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