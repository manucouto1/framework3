from typing import Any, Optional
from framework3.base.base_types import XYData, VData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container
from framework3.base.base_types import ArrayLike
from sklearn.linear_model import LogisticRegression


Container.bind()
class LogistiRegressionlugin(BaseFilter, BasePlugin):
    def __init__(self, max_ite: int, tol: float):
        self._logistic = LogisticRegression(max_iter=max_ite, tol=tol)
    
    def fit(self, x:XYData, y:XYData|None) -> None:
        if y is not None and type(y) == ArrayLike:
            self._logistic.fit(x._value, y._value)  # type: ignore
        else:
            raise ValueError("y must be provided for logistic regression")

    def predict(self, x:XYData) -> VData:
        return self._logistic.predict(x.value)