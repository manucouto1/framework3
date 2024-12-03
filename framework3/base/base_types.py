from __future__ import annotations
from typing import Callable, Generic, Iterable, Literal, Sequence, TypeVar, Any, cast
import pandas as pd
import numpy as np
import torch
import typing_extensions
from multimethod import multimethod
from scipy.sparse import spmatrix, hstack, vstack
from typing import TypeAlias

from dataclasses import dataclass, field

__all__ = ["XYData", "VData", "SkVData", "IncEx", "TypePlugable"]

Float = float | np.float16 | np.float32 | np.float64
IncEx: typing_extensions.TypeAlias = (
    "set[int] | set[str] | dict[int, Any] | dict[str, Any] | None"
)

VData: TypeAlias = np.ndarray | pd.DataFrame | spmatrix | list | torch.Tensor
SkVData: TypeAlias = np.ndarray | pd.DataFrame | spmatrix

TypePlugable = TypeVar("TypePlugable")
TxyData = TypeVar("TxyData", SkVData, VData)


@dataclass(slots=True, frozen=True)
class XYData(Generic[TxyData]):
    """
    A dataclass representing data for machine learning tasks, typically features (X) or targets (Y).

    This class is immutable (frozen) and uses slots for memory efficiency.

    Attributes:
        _hash (str): A unique identifier or hash for the data.
        _path (str): The path where the data is stored or retrieved from.
        _value (VData | Callable[..., VData]): The actual data or a callable that returns the data.
                                               It can be a numpy array, pandas DataFrame, or scipy sparse matrix.
    """

    _hash: str = field(init=True)
    _path: str = field(init=True)
    _value: TxyData | Callable[..., TxyData] = field(init=True, repr=False)

    @staticmethod
    def mock(value: TxyData | Callable[..., TxyData]) -> XYData:
        """
        Create a mock XYData instance for testing or placeholder purposes.

        Args:
            value (VData | Callable[..., VData]): The data or callable to use for the mock instance.

        Returns:
            XYData: A new XYData instance with mock values.

        Example:
            ```python

            >>> mock_data = XYData.mock(np.random.rand(10, 5))
            >>> mock_data.value.shape
            (10, 5)
            ```

        """
        return XYData(_hash="Mock", _path="", _value=value)

    @property
    def value(self) -> TxyData:
        """
        Property to access the actual data.

        If _value is a callable, it will be called to retrieve the data.
        Otherwise, it returns the data directly.

        Returns:
            VData: The actual data (numpy array, pandas DataFrame, or scipy sparse matrix).
        """
        return self._value() if callable(self._value) else self._value

    @staticmethod
    def concat(x: Sequence[TxyData], axis: int = -1) -> XYData:
        return concat(x, axis=axis)

    @staticmethod
    def ensure_dim(x: list | np.ndarray) -> list | np.ndarray:
        return ensure_dim(x)

    def as_iterable(self) -> Iterable:
        """
        Convert the `_value` attribute to an iterable, regardless of its underlying type.

        Returns:
            Iterable: An iterable version of `_value`.
        """
        value = self.value

        # Maneja diferentes tipos de datos
        if isinstance(value, np.ndarray):
            return value  # Los arrays numpy ya son iterables
        elif isinstance(value, pd.DataFrame):
            return value.iterrows()  # Devuelve un iterable sobre las filas
        elif isinstance(value, spmatrix):
            return value.toarray()  # type: ignore # Convierte la matriz dispersa a un array denso
        else:
            raise TypeError(f"El tipo {type(value)} no es compatible con iteración.")


@multimethod
def concat(x: Any, axis: int) -> "XYData":
    raise TypeError(f"Cannot concatenate this type of data, only {VData} compatible")


@concat.register  # type: ignore
def _(x: list[np.ndarray], axis: int = -1) -> "XYData":
    return XYData.mock(np.concatenate(x, axis=axis))


@concat.register  # type: ignore
def _(x: list[pd.DataFrame], axis: int = -1) -> "XYData":
    return XYData.mock(pd.concat(x, axis=axis))  # type: ignore


@concat.register  # type: ignore
def _(x: list[spmatrix], axis: Literal[0, 1, -1]) -> "XYData":
    if axis == 1:
        return XYData.mock(value=cast(spmatrix, hstack(x)))
    elif axis == 0:
        return XYData.mock(value=cast(spmatrix, vstack(x)))
    raise ValueError("Invalid axis for concatenating sparse matrices")


@multimethod
def ensure_dim(x: Any) -> SkVData:
    raise TypeError(f"Cannot concatenate this type of data, only {VData} compatible")


@ensure_dim.register  # type: ignore
def _(x: np.ndarray) -> SkVData:
    if x.ndim == 1:  # Verifica si es unidimensional
        return x[:, None]  # Agrega una nueva dimensión
    return x  # No cambia el array si tiene más dimensiones


@ensure_dim.register  # type: ignore
def _(x: list) -> SkVData:
    return ensure_dim(np.array(x))
