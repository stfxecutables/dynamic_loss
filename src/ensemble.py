from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import torch
from pandas import DataFrame, Index
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from torchmetrics.functional import accuracy
from tqdm import tqdm
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
        agg_logits = np.nanmean(preds, axis=0)  # shape is (n_samples, n_classes)
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


def summarize_ensemble_base_accs() -> DataFrame:
    outdir = RESULTS / "ensemble_base_stats"
    outdir = outdir / "static"
    outdir = outdir / "each"
    outdir.mkdir(exist_ok=True, parents=True)
    finished = datetime.utcnow().strftime("%Y.%m.%d--%H-%M-%S.%f")
    outfile = outdir / f"{finished}.json"

    dfs = []
    for threshold in [None, 0.6, 0.7, 0.8, 0.9]:
        for ds in [
            VisionDataset.CIFAR10,
            VisionDataset.CIFAR100,
            VisionDataset.FashionMNIST,
        ]:
            preds, targs, idxs = consolidate_preds(
                ds, phase=FinalEvalPhase.Test, threshold=threshold
            )

            nan_ensembles = np.unique(np.where(np.isnan(preds))[0]).ravel().tolist()
            preds = np.delete(preds, nan_ensembles, axis=0)
            targs = np.delete(targs, nan_ensembles, axis=0)

            votes = np.argmax(preds, axis=-1)  # votes[:, i] are labels for sample i
            agg_logits = np.nanmean(preds, axis=0)  # shape is (n_samples, n_classes)
            aggs = np.argmax(agg_logits, axis=1)

            ensemble_accs = np.nanmean(votes == targs, axis=1)  # (n_ensembles,)
            df_accs = DataFrame(ensemble_accs).describe().T
            df_accs.index = [0]
            df_data = DataFrame({"data": ds.name}, index=[0])
            df = DataFrame(
                {"thresh": threshold if threshold is not None else -1}, index=[0]
            )
            df = pd.concat([df_data, df, df_accs], axis=1, ignore_index=False)
            dfs.append(df)
            continue

            # sd_acc_vote = np.std(ensemble_accs, ddof=1)
            # acc_min_vote, acc_max_vote = ensemble_accs.min(), ensemble_accs.max()
            # acc_avg_vote = np.mean(ensemble_accs)

            # all_accs_agg = np.mean(aggs == targs[0], axis=1)  # (n_ensembles,)
            # sd_acc_agg = np.std(all_accs_agg, ddof=1)
            # acc_min_agg, acc_max_agg = all_accs_agg.min(), all_accs_agg.max()
            # acc_avg_agg = np.mean(all_accs_agg)

            # if config.fusion is FusionMethod.Vote:
            #     votes = np.argmax(preds, axis=-1)  # votes[:, i] are labels for sample i
            #     vote_maxs = np.max(preds, axis=0)  # may need to softmax this

            #     vote_preds = mode(votes, axis=0)[0].squeeze()
            #     acc = np.mean(vote_preds == targ)
            #     top3 = accuracy(
            #         torch.from_numpy(vote_maxs), torch.from_numpy(targ), top_k=3
            #     ).item()
            #     top5 = accuracy(
            #         torch.from_numpy(vote_maxs), torch.from_numpy(targ), top_k=5
            #     ).item()
            # elif config.fusion is FusionMethod.Average:
            #     agg_logits = np.nanmean(preds, axis=0)  # shape is (n_samples, n_classes)
            #     agg_preds = np.argmax(agg_logits, axis=1)
            #     acc = np.mean(agg_preds == targ)
            #     top3 = accuracy(
            #         torch.from_numpy(agg_logits), torch.from_numpy(targ), top_k=3
            #     ).item()
            #     top5 = accuracy(
            #         torch.from_numpy(agg_logits), torch.from_numpy(targ), top_k=5
            #     ).item()
            # else:
            #     raise ValueError("Fusion method not a classic fusion method")
    df = (
        pd.concat(dfs, axis=0, ignore_index=False)
        .drop(columns=["count", "25%", "50%", "75%"])
        .sort_values(by=["data", "thresh"])
    )
    return df.pivot(index=["data", "thresh"], columns=[])


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


def fusion_sorter(s: str) -> str:
    if s == "Average":
        return "A"
    if s == "Vote":
        return "B"
    if s == "Weighted":
        return "C"
    if s == "MLP":
        return "D"


def print_all_ensemble_accs_everything() -> None:
    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000

    outdir = RESULTS / "ensemble_evals"
    paths = sorted(outdir.rglob("*.json"))
    dfs = [pd.read_json(js) for js in tqdm(paths, desc="Loading")]
    df = pd.concat(dfs, axis=0)

    topk_drops = [c for c in df.columns if "top" in c]
    loss_drops = [c for c in df.columns if "loss" in c]
    val_cols = [c for c in df.columns if "val" in c]
    df = df.drop(columns=topk_drops + loss_drops + val_cols)
    print(
        df.sort_values(by=["fusion", "data", "thresh", "wd"])
        .drop(
            columns=["pooled", "shuffled", "meta_epochs_l", "meta_epochs_b", "acc_test_l"]
        )
        .to_markdown(index=False, tablefmt="simple")
    )
    data = df.sort_values(by=["fusion", "data", "thresh", "wd"]).drop(
        columns=["pooled", "shuffled", "meta_epochs_l", "meta_epochs_b", "acc_test_l"]
    )
    data = data[data.fusion.isin(["MLP", "Weighted"])]
    data = data[data["acc_test_b"] > 0.2]
    data.thresh = data.thresh.apply(lambda x: 0 if x < 0 else x)
    sbn.catplot(
        data=data,
        x="wd",
        y="acc_test_b",
        row="fusion",
        col="data",
        kind="strip",
        hue="thresh",
    )
    plt.show()

    learn_cols = ["lr", "wd", "meta_epochs_l", "meta_epochs_b"]
    df = df[(df.wd == 0) | (df.fusion.isin(["Average", "Vote"]))]
    df = df.drop(columns=learn_cols)

    pooled = df["pooled"] == True  # these were always worse
    df = df.loc[~pooled, :]
    df = df.drop(columns="pooled")

    shuffled = df["shuffled"] == True  # these were also always worse
    df = df.loc[~shuffled, :]
    df = df.drop(columns="shuffled")

    df = df.drop(columns="acc_test_l")  # generally same as best, missing for static
    df = df.rename(columns={"acc_test_b": "acc"})
    botched_nan_runs = df["acc"] < 0.2
    df = df.loc[~botched_nan_runs]

    df["fsort"] = df.fusion.apply(fusion_sorter)

    print(
        df.sort_values(by=["data", "fsort", "thresh"])
        .drop(columns="fsort")
        .to_markdown(index=False, tablefmt="simple")
    )
    # df["fusion_kind"] = df.fusion.apply(
    #     lambda f: "static" if f in ["Vote", "Average"] else "learned"
    # )
    pivoted = (
        df.sort_values(by=["data", "fusion", "fsort"])
        .drop(columns="fsort")
        .pivot(index="thresh", columns=["data", "fusion"])
    )
    col_order = ["Average", "Vote", "Weighted", "MLP"]
    pivoted = pivoted.reindex(columns=col_order, level=2)
    print("Effect of dynamic loss threshold:")
    print(pivoted)

    print("\nThreshold test-accuracy correlations")
    corrs = (
        df.groupby(["data", "fusion", "fsort"])
        .corr()["thresh"]
        .reset_index()
        .iloc[slice(1, None, 2)]
        .drop(columns="level_3")
        .sort_values(by=["data", "fsort"])
        .drop(columns="fsort")
        .pivot(index="fusion", columns="data")
    )
    print(corrs.round(3))

    # print(df.groupby(["data", "fusion"]).corr(method="spearman"))
    return

    cols = ["data", "fusion", "thresh", "acc_test_b", "top3_test_b", "top5_test_b"]
    df = df.loc[:, cols]
    df = df.rename(columns=lambda s: s.replace("test_b", "test")).sort_values(
        by=["data", "fusion", "thresh"]
    )
    print(df.to_markdown(index=False, floatfmt="0.4f", tablefmt="simple"))


if __name__ == "__main__":
    summarize_ensemble_base_accs()
    # print_all_ensemble_accs_everything()
    # print_all_classic_ensemble_accs()
    # summarize_all_classic_ensemble_accs()
    # gradboost(VisionDataset.CIFAR100)
    # knnfit(VisionDataset.CIFAR100)
    # svcfit(VisionDataset.CIFAR100)
    # svcfit(VisionDataset.CIFAR100)
