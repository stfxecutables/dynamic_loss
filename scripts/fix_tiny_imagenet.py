from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from shutil import copyfile, move

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.constants import DATA

TINY = DATA / "tiny_imagenet"
TINY_ROOT = TINY / "tiny-imagenet-200"
TRAIN_START = TINY_ROOT / "train"
VAL_START = TINY_ROOT / "val"
TEST_START = TINY_ROOT / "test"
STARTS = [TRAIN_START, VAL_START, TEST_START]

TRAIN = TINY / "train"
VAL = TINY / "val"
TEST = TINY / "test"  # Actually useless, no labels
ENDS = [TRAIN, VAL, TEST]

VAL_ANNOTATIONS = VAL / "val_annotations.txt"
VAL_IMAGES = VAL / "images"

X_TRAIN_OUT = DATA / "tiny_x_train.npz"
Y_TRAIN_OUT = DATA / "tiny_y_train.npz"
X_TEST_OUT = DATA / "tiny_x_test.npz"
Y_TEST_OUT = DATA / "tiny_y_test.npz"

# Entire train + val images are 64x64x3 uint8 representable, 110 000 images, so
# entire dataset can be loaded into memory for just about 10GiB. We do that
# instead of keeping in worthless .JPEG form which will be unusable on Cedar


def move_files() -> None:
    if TINY_ROOT.exists():
        for start_dir, end_dir in zip(STARTS, ENDS):
            if start_dir.exists():
                move(start_dir, end_dir)
        text_files = sorted(TINY_ROOT.glob("*.txt"))
        for text in text_files:
            move(text, TINY)
        try:
            TINY_ROOT.rmdir()
        except Exception as e:
            raise FileExistsError(
                f"Could not delete non-empty directory: {TINY_ROOT}"
            ) from e


def organize_val_images() -> None:
    with open(VAL_ANNOTATIONS, "r") as handle:
        lines = handle.readlines()

    clean = [line.replace("\n", "").split("\t")[:2] for line in lines]
    imgs, labels = zip(*clean)
    unq_labels = np.unique(labels).tolist()
    for unq in unq_labels:
        (VAL_IMAGES / unq).mkdir(exist_ok=True, parents=True)

    for img, label in tqdm(
        zip(imgs, labels), total=len(imgs), desc="Organizing val images"
    ):
        img_start = VAL_IMAGES / img
        img_end = VAL_IMAGES / f"{label}/{img}"
        if not img_end.exists():
            move(img_start, img_end)


def to_numpy() -> None:
    train_paths = sorted((TINY / "train").rglob("*.JPEG"))
    val_paths = sorted((TINY / "val").rglob("*.JPEG"))
    labels = [p.parent.parent.name for p in train_paths]
    unq_labels = np.unique(labels).tolist()
    legend = {lab: i for i, lab in enumerate(unq_labels)}

    X_train, y_train = [], []
    for path in tqdm(train_paths, desc="Converting train images"):
        x = np.asarray(Image.open(path), dtype=np.uint8)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=0)
        else:
            x = x.transpose(2, 0, 1)
        y = legend[path.parent.parent.name]
        X_train.append(x)
        y_train.append(y)
    X_test, y_test = [], []
    for path in tqdm(val_paths, desc="Converting val images"):
        x = np.asarray(Image.open(path), dtype=np.uint8)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=0)
        else:
            x = x.transpose(2, 0, 1)
        y = legend[path.parent.name]
        X_test.append(x)
        y_test.append(y)

    X_train = np.stack(X_train)
    np.savez_compressed(X_TRAIN_OUT, X_train=X_train)
    y_train = np.stack(y_train)
    np.savez_compressed(Y_TRAIN_OUT, y_train=y_train)
    X_test = np.stack(X_test)
    np.savez_compressed(X_TEST_OUT, X_test=X_test)
    y_test = np.stack(y_test)
    np.savez_compressed(Y_TEST_OUT, y_test=y_test)


if __name__ == "__main__":
    move_files()
    organize_val_images()
    to_numpy()
