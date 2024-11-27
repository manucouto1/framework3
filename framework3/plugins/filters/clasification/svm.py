from ast import Tuple
from typing import Any, Dict, List, Literal, Type, cast
from framework3.base.base_types import XYData, ArrayLike, MatrixLike
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container
from sklearn.svm import SVC

L = Literal['linear', 'poly', 'rbf', 'sigmoid']
@Container.bind()
class ClassifierSVMPlugin(BaseFilter, BasePlugin):
    def __init__(self, C:float=1.0, gamma:float|Literal['scale', 'auto']='scale', kernel:L='linear') -> None:
        super().__init__(C=C, kernel=kernel, gamma=gamma)
        self._model = SVC(C=C, kernel=kernel, gamma=gamma) 
    
    def fit(self, x:XYData, y:XYData|None):
        if y is not None :
            self._model.fit(x.value, y.value)  # type: ignore
    
    def predict(self, x:XYData):
        return self._model.predict(x.value)
    
    @staticmethod
    def item_grid(C: List[float], kernel: List[L], gamma: List[float]|List[Literal['scale', 'auto']]=['scale']) -> Dict[str, Any]:
        return {"filterx":ClassifierSVMPlugin, "param_grid":{'C': C, 'kernel': kernel, 'gamma': gamma}}
