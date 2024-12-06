import pytest
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from typing import Optional

from framework3.base.exceptions import NotTrainableFilterError


class TrainableFilter(BaseFilter):
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        self.is_fitted = True

    def predict(self, x: XYData) -> XYData:
        if not self.is_fitted:
            raise ValueError("This filter needs to be fitted before prediction")
        return x


class NonTrainableFilter(BaseFilter):
    def __init__(self):
        self.init()

    def predict(self, x: XYData) -> XYData:
        return x


def test_trainable_filter():
    filter = TrainableFilter()
    x = XYData.mock([1, 2, 3])

    # Verificar que el filtro no está inicializado
    assert not hasattr(filter, "_m_hash")
    assert not hasattr(filter, "_m_str")
    assert not hasattr(filter, "_m_path")

    # Intentar predecir sin entrenar debería fallar
    with pytest.raises(ValueError):
        filter.predict(x)

    # Entrenar el filtro
    filter.fit(x, None)

    # Verificar que el filtro está inicializado después del entrenamiento
    assert hasattr(filter, "_m_hash")
    assert hasattr(filter, "_m_str")
    assert hasattr(filter, "_m_path")

    # La predicción ahora debería funcionar
    result = filter.predict(x)
    assert isinstance(result, XYData)


def test_non_trainable_filter():
    filter = NonTrainableFilter()
    x = XYData.mock([1, 2, 3])

    # Verificar que el filtro está inicializado desde el principio
    assert hasattr(filter, "_m_hash")
    assert hasattr(filter, "_m_str")
    assert hasattr(filter, "_m_path")

    # La predicción debería funcionar sin necesidad de entrenamiento
    result = filter.predict(x)
    assert isinstance(result, XYData)

    # El método fit no debería cambiar el estado de inicialización
    initial_hash = filter._m_hash

    with pytest.raises(NotTrainableFilterError):
        filter.fit(x, None)

    assert filter._m_hash == initial_hash
