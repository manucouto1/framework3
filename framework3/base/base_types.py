from __future__ import annotations
from typing import Callable, Protocol, Tuple, TypeVar, Generic, List, Dict, Union, Any, Sequence, Optional, cast
from pydantic._internal._model_construction import ModelMetaclass
import pandas as pd
import numpy as np
import torch
import typing_extensions
import decimal
import io
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix


Decimal = decimal.Decimal
PythonScalar = str | int | float | bool

MatrixLike = np.ndarray | pd.DataFrame | spmatrix
FileLike = io.IOBase
PathLike = str
Int = int | np.int8 | np.int16 | np.int32 | np.int64
Float = float | np.float16 | np.float32 | np.float64

PandasScalar = pd.Period | pd.Timestamp | pd.Timedelta | pd.Interval
Scalar = PythonScalar | PandasScalar
XYDataG = Union[List[Any], np.ndarray, torch.Tensor, pd.DataFrame]

# XYData = np.ndarray | pd.DataFrame | spmatrix

IncEx: typing_extensions.TypeAlias = 'set[int] | set[str] | dict[int, Any] | dict[str, Any] | None'

PyDanticMeta = ModelMetaclass

TypePlugable = TypeVar('TypePlugable')


from dataclasses import dataclass, field

VData = np.ndarray | pd.DataFrame | spmatrix

@dataclass(slots=True, frozen=True)
class XYData:
    _hash: str = field(init=True)
    _path: str = field(init=True)
    _value: VData|Callable[...,VData] = field(init=True, repr=False)

    @staticmethod
    def mock(value: VData|Callable[...,VData]) -> XYData:
        return XYData(
            _hash=f"Mock",
            _path='',
            _value=value
        )

    @property
    def value(self) -> VData:
        return  self._value() if callable(self._value) else self._value

    # @hash.setter
    # def hash(self, value: str):
    #     if not isinstance(value, str):
    #         raise ValueError("Hash must be a string")
    #     object.__setattr__(self, '_hash', value)