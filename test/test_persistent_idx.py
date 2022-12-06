from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

# derived from os.urandom(64) at some point
SEED = int.from_bytes(
    b"\xfe\xeb$\xd17~s\xd5\xea\x0ba\xc4\x9d\xb4y\xf9\xf8\x92de\xbb\x0bg\xf5\xb8\xe9\x16\x9aL\x17\x96\x9fk\x94\x14\xf9\xd1\xca\\\t\xa9N\x0f\xa8\t\xdbE\xa0\xb8\x86&\xc8\xb1\xb5\x8d9,h\xd8\x17Q5_\x7f",
    byteorder="big",
)


def get_shuffle_idx(n_indices: int, size: int) -> list[ndarray]:
    ss = np.random.SeedSequence(entropy=SEED)
    seeds = ss.spawn(n_indices)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    idxs = [rng.permutation(size) for rng in rngs]
    return idxs


def test_persistent(n_indices: int, size: int) -> None:
    idxs1 = np.stack(get_shuffle_idx(n_indices=n_indices, size=size))
    idxs2 = np.stack(get_shuffle_idx(n_indices=n_indices, size=size))
    np.testing.assert_array_equal(idxs1, idxs2)

    prev = Path(__file__).resolve().parent / f"prev_{n_indices}_{size}.npy"
    if not prev.exists():
        np.save(prev, idxs1)
    idxs_prev = np.load(prev)
    np.testing.assert_array_equal(idxs1, idxs_prev)
    np.testing.assert_array_equal(idxs2, idxs_prev)


if __name__ == "__main__":
    for n_indices in [5, 10, 50]:
        for size in [5, 500, 50000]:
            test_persistent(n_indices=n_indices, size=size)
            test_persistent(n_indices=n_indices, size=size)
