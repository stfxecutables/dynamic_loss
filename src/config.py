from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from dataclasses import dataclass


@dataclass
class Config:
    max_epochs: int = 30
    lr_init: float = 5e-4
    weight_decay: float = 5e-3