from __future__ import annotations
from typing import Callable, TypeVar, List, Any
from pydantic._internal._model_construction import ModelMetaclass
import pandas as pd
import numpy as np
import typing_extensions
from scipy.sparse import spmatrix

from dataclasses import dataclass, field

__all__=  ["XYData", "VData", "IncEx", "TypePlugable"]

Float = float | np.float16 | np.float32 | np.float64
IncEx: typing_extensions.TypeAlias = 'set[int] | set[str] | dict[int, Any] | dict[str, Any] | None'
TypePlugable = TypeVar('TypePlugable')
VData = np.ndarray | pd.DataFrame | spmatrix

@dataclass(slots=True, frozen=True)
class XYData:
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
    _value: VData | Callable[..., VData] = field(init=True, repr=False)

    @staticmethod
    def mock(value: VData | Callable[..., VData]) -> XYData:
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
        return XYData(
            _hash="Mock",
            _path='',
            _value=value
        )

    @property
    def value(self) -> VData:
        """
        Property to access the actual data.

        If _value is a callable, it will be called to retrieve the data.
        Otherwise, it returns the data directly.

        Returns:
            VData: The actual data (numpy array, pandas DataFrame, or scipy sparse matrix).
        """
        return self._value() if callable(self._value) else self._value