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
from xgboost import XGBClassifier
from xgboost.callback import EvaluationMonitor

from src.config import Config
from src.constants import CC_LOGS, LOG_ROOT_DIR, ON_CCANADA, RESULTS
from src.enumerables import FinalEvalPhase, FusionMethod, VisionDataset
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


def summarize_ensemble_accs(config: Config, threshold: float | None) -> DataFrame:
    outdir = RESULTS / "ensemble_evals"
    outdir = outdir / "static"
    outdir = outdir / "each"
    outdir.mkdir(exist_ok=True, parents=True)
    finished = datetime.utcnow().strftime("%Y.%m.%d--%H-%M-%S.%f")
    outfile = outdir / f"{finished}.json"

    preds, targs, idxs = consolidate_preds(
        config.vision_dataset, phase=FinalEvalPhase.Test, threshold=threshold
    )
    targ = targs[0]

    if config.fusion is FusionMethod.Vote:
        votes = np.argmax(preds, axis=-1)  # votes[:, i] are labels for sample i
        vote_maxs = np.max(preds, axis=0)  # may need to softmax this

        vote_preds = mode(votes, axis=0)[0].squeeze()
        acc = np.mean(vote_preds == targ)
        top3 = accuracy(
            torch.from_numpy(vote_maxs), torch.from_numpy(targ), top_k=3
        ).item()
        top5 = accuracy(
            torch.from_numpy(vote_maxs), torch.from_numpy(targ), top_k=5
        ).item()
    elif config.fusion is FusionMethod.Average:
        agg_logits = np.mean(preds, axis=0)  # shape is (n_samples, n_classes)
        agg_preds = np.argmax(agg_logits, axis=1)
        acc = np.mean(agg_preds == targ)
        top3 = accuracy(
            torch.from_numpy(agg_logits), torch.from_numpy(targ), top_k=3
        ).item()
        top5 = accuracy(
            torch.from_numpy(agg_logits), torch.from_numpy(targ), top_k=5
        ).item()
    else:
        raise ValueError("Fusion method not a classic fusion method")

    # all_accs = np.mean(votes == targs, axis=1)  # (n_ensembles,)
    # sd_acc = np.std(all_accs, ddof=1)
    # acc_min, acc_max = all_accs.min(), all_accs.max()
    # acc_avg = np.mean(all_accs)

    df = DataFrame(
        {
            "data": config.vision_dataset.name,
            "fusion": config.fusion.name,
            "pooled": False,
            "shuffled": False,
            "thresh": threshold if threshold is not None else -1,
            "acc_test_l": -1,
            "top3_test_l": -1,
            "top5_test_l": -1,
            "loss_test_l": -1,
            "acc_val_l": -1,
            "top3_val_l": -1,
            "top5_val_l": -1,
            "loss_val_l": -1,
            "acc_test_b": acc,
            "top3_test_b": top3,
            "top5_test_b": top5,
            "loss_test_b": -1,
            "acc_val_b": -1,
            "top3_val_b": -1,
            "top5_val_b": -1,
            "loss_val_b": -1,
            "meta_epochs_l": -1,
            "meta_epochs_b": -1,
            "lr": config.lr_init,
            "wd": config.weight_decay,
        },
        index=[finished],
    )
    df.to_json(outfile)
    print(f"Saved results to {outfile}")
    return df


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


def knnfit(dataset: VisionDataset) -> None:
    # only gets like 60% on CIFAR-100
    X_train, y_train, X_test, y_test = ml_data(dataset)
    knn = KNN(n_neighbors=1, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test)}")


def svcfit(dataset: VisionDataset) -> None:
    # only gets like 60% on CIFAR-100
    X_train, y_train, X_test, y_test = ml_data(dataset)
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(f"SVM Accuracy: {np.mean(y_pred == y_test)}")


def summarize_all_classic_ensemble_accs() -> None:
    for ds in [VisionDataset.CIFAR10, VisionDataset.CIFAR100, VisionDataset.FashionMNIST]:
        for fusion in [FusionMethod.Vote, FusionMethod.Average]:
            for threshold in [None, 0.6, 0.7, 0.8, 0.9]:
                argstr = (
                    "--experiment=ensemble-eval "
                    "--subset=full "
                    f"--dataset={ds.value} "
                    f"--fusion={fusion.value} "
                    f"--max_epochs=1"
                )
                print(f"Evaluating with args:")
                print(argstr)
                config = Config.from_args(argstr=argstr)[0]
                summarize_ensemble_accs(config=config, threshold=threshold)


def print_all_classic_ensemble_accs() -> None:
    outdir = RESULTS / "ensemble_evals/static/each"
    paths = outdir.rglob("*.json")
    dfs = [pd.read_json(js) for js in paths]
    df = pd.concat(dfs, axis=0)
    cols = ["data", "fusion", "thresh", "acc_test_b", "top3_test_b", "top5_test_b"]
    df = df.loc[:, cols]
    df = df.rename(columns=lambda s: s.replace("test_b", "test")).sort_values(
        by=["data", "fusion", "thresh"]
    )
    print(df.to_markdown(index=False, floatfmt="0.4f", tablefmt="simple"))


if __name__ == "__main__":
    print_all_classic_ensemble_accs()
    # gradboost(VisionDataset.CIFAR100)
    # knnfit(VisionDataset.CIFAR100)
    # svcfit(VisionDataset.CIFAR100)
