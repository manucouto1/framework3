from ast import Tuple
from typing import Any, Dict, List, Literal, Type
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
    
    def fit(self, x, y:Any):
        if y is not None:
            self._model.fit(x, y) 
    
    def predict(self, x:MatrixLike ):
        return self._model.predict(x)
    
    @staticmethod
    def item_grid(C: List[float], kernel: List[L], gamma: List[float]|List[Literal['scale', 'auto']]=['scale']) -> Dict[str, Any]:
        return {"filterx":ClassifierSVMPlugin, "param_grid":{'C': C, 'kernel': kernel, 'gamma': gamma}}
