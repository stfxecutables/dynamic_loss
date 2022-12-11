from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def callbacks(log_version_dir: Path) -> list[Callback]:
    return [
        # LearningRateMonitor(logging_interval="epoch"),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=log_version_dir / "ckpts",
            filename="epoch={epoch}-val_acc={val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            auto_insert_metric_name=False,
            save_last=True,
            save_top_k=1,
            save_weights_only=False,
        ),
        # Only to watch for NaN
        EarlyStopping(
            monitor="val/acc",
            patience=200,
            mode="max",
            check_finite=True
            # divergence_threshold=-4.0,
        ),
    ]


def ensemble_callbacks(log_version_dir: Path, max_epochs: int) -> list[Callback]:
    return [
        ModelCheckpoint(
            dirpath=log_version_dir / "ckpts",
            filename="epoch={epoch}-val_acc={val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            auto_insert_metric_name=False,
            save_last=True,
            save_top_k=1,
            save_weights_only=False,
        ),
        EarlyStopping(
            monitor="val/acc",
            patience=max(5, max_epochs // 5),
            mode="max",
            check_finite=True
            # divergence_threshold=-4.0,
        ),
    ]
