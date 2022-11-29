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
    evaluate(tune=True)

