from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from torchmetrics.functional import accuracy
from tqdm import tqdm

from src.config import Config
from src.constants import CC_LOGS, LOG_ROOT_DIR, ON_CCANADA, RESULTS
from src.ensemble import print_all_ensemble_accs_everything
from src.enumerables import FinalEvalPhase, FusionMethod, VisionDataset
from src.loaders.ensemble import ml_data
from src.loaders.preds import consolidate_preds

LOG_DIR = LOG_ROOT_DIR if ON_CCANADA else CC_LOGS / "logs"
BOOT_DIR = LOG_DIR / "base-train/boot"

if __name__ == "__main__":
    print_all_ensemble_accs_everything()