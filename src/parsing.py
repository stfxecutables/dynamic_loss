from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from enum import EnumMeta
from typing import Callable, Optional, Type, TypeVar, Union

T = TypeVar("T")


def to_bool(arg: str) -> bool:
    orig = arg
    arg = arg.lower()
    if arg in ["true", "t", "1"]:
        return True
    if arg in ["false", "f", "0"]:
        return False
    raise ValueError(f"`{orig}` is not valid for a boolean argument.")


def int_or_none(s: str) -> Optional[int]:
    if s.lower() == "none":
        return None
    return int(s)


def float_or_none(s: str) -> Optional[int]:
    if s.lower() == "none":
        return None
    return float(s)


def get_parser(typ: Type[T]) -> Callable[[str], T]:
    if typ is bool:
        return to_bool
    elif typ is int:
        return int
    elif typ is float:
        return float
    elif isinstance(typ, EnumMeta):
        return typ
    elif typ is str:
        return str
    else:
        raise ValueError(f"No parser defined for arguments of type: {typ}")
