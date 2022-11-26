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


class Phase(Enum):
    Train = "train"
    Val = "val"
    Pred = "pred"
    Test = "test"