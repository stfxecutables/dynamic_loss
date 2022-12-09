from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import mode
from xgboost import XGBClassifier
from xgboost.callback import EvaluationMonitor

from src.constants import CC_LOGS, LOG_ROOT_DIR, ON_CCANADA
from src.enumerables import FinalEvalPhase, VisionDataset
from src.loaders.ensemble import ml_data
from src.loaders.preds import consolidate_preds

LOG_DIR = LOG_ROOT_DIR if ON_CCANADA else CC_LOGS / "logs"
BOOT_DIR = LOG_DIR / "base-train/boot"


def summarize_classic_ensemble_test_accs(dataset: VisionDataset) -> DataFrame:
    preds, targs, idxs = consolidate_preds(dataset, FinalEvalPhase.Test)
    targ = targs[0]
    votes = np.argmax(preds, axis=-1)  # votes[:, i] are labels for sample i
    vote_preds = mode(votes, axis=0)[0].squeeze()
    agg_logits = np.mean(preds, axis=0)  # shape is (n_samples, n_classes)
    agg_preds = np.argmax(agg_logits, axis=1)

    all_accs = np.mean(votes == targs, axis=1)  # (n_ensembles,)
    sd_acc = np.std(all_accs, ddof=1)
    acc_min, acc_max = all_accs.min(), all_accs.max()
    acc_avg = np.mean(all_accs)

    vote_acc = np.mean(vote_preds == targ)
    agg_acc = np.mean(agg_preds == targ)
    return DataFrame(
        {
            "data": dataset.value,
            "base_avg": acc_avg,
            "base_sd": sd_acc,
            "base_min": acc_min,
            "base_max": acc_max,
            "vote": vote_acc,
            "agg": agg_acc,
        },
        index=[dataset.value],
    )


def print_all_classic_results() -> None:
    dfs = []
    for ds in [VisionDataset.FashionMNIST, VisionDataset.CIFAR10, VisionDataset.CIFAR100]:
        dfs.append(summarize_classic_ensemble_test_accs(ds))
    df = pd.concat(dfs, axis=0)
    print(df.round(4).to_markdown(tablefmt="simple", index=False))


def gradboost(dataset: VisionDataset) -> None:
    X_train, y_train, X_test, y_test = ml_data(dataset)
    xgb = XGBClassifier(
        n_estimators=250, n_jobs=-1, callbacks=[EvaluationMonitor()], tree_method="hist"
    )
    xgb.fit(X_train, y_train, verbose=True)
    y_pred = xgb.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test)}")


if __name__ == "__main__":
    # print_all_classic_results()
    gradboost(VisionDataset.CIFAR100)
