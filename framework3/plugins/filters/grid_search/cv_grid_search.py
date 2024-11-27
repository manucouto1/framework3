from typing import Any, Dict, List, Type

from sklearn.base import BaseEstimator
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container
from sklearn.model_selection import GridSearchCV

from framework3.container.model.bind_model import BindGenericModel

class SkFilterWrapper(BaseEstimator):
    z_clazz: Type[BaseFilter]
    def __init__(self, **kwargs):
        self._model = SkFilterWrapper.z_clazz(**kwargs)
        self.kwargs = kwargs
    
    def fit(self, x,y, *args, **kwargs):
        
        self._model.fit(XYData.mock(x), XYData.mock(y))
        return self

    def predict(self, x):
        return self._model.predict(XYData.mock(x))
    
    # def score(self, data):
    #     return self.model.score(data)

    def get_params(self, deep=True):
        return {**self.kwargs}

    def set_params(self, **parameters):
        self._model = SkFilterWrapper.z_clazz(**parameters)
        return self

@Container.bind()
class GridSearchCVPlugin(BaseFilter):
    def __init__(self, filterx:  Type[BaseFilter], param_grid: Dict[str, Any], scoring:str, cv:int=2): #TODO primero implemento con un solo filtro
        super().__init__(filterx=filterx, param_grid=param_grid, scoring=scoring, cv=cv)
        # SkFilterWrapper.z_clazz = Container.ff[filterx]

        SkFilterWrapper.z_clazz = filterx
        self._clf:GridSearchCV = GridSearchCV(
            estimator=SkFilterWrapper(), 
            param_grid=param_grid, 
            scoring=scoring, 
            cv=cv, 
            refit=True,
            verbose=0
        )

    def fit(self, x, y):
        self._clf.fit(x.value, y.value) # type: ignore
    
    def predict(self, x ):
        return self._clf.predict(x.value) # type: ignore