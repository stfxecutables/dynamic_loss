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
        copyfile(img_start, img_end)


if __name__ == "__main__":
    move_files()
    organize_val_images()
