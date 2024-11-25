from typing import Any, Dict, List, Optional, Type
import pytest
import numpy as np
from sklearn.base import BaseEstimator

from framework3.container.container import Container
from framework3.base.base_clases import BasePlugin, BaseFilter
from framework3.base.base_types import XYData
from framework3.container.model.bind_model import BindGenericModel
from framework3.plugins.filters.grid_search.cv_grid_search import GridSearchCVPlugin


class DummyFilter(BaseFilter):
    def __init__(self, param1: int|List[int], param2: float|List[float]):
        ...
    
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        pass

    def predict(self, x: XYData) -> XYData:
        return x
    
# try:

class SkFilterWrapper(BasePlugin):
    def __init__(self, clazz: Type[BaseFilter], **kwargs):
        self.model = clazz(**kwargs)
        self.z_clazz = clazz
        self.kwargs = kwargs
    
    def fit(self, data):
        print(data)
        self.model.fit(data[0], data[1])
        return self

    def predict(self, data):
        return self.model.predict(data)
    
    # def score(self, data):
    #     return self.model.score(data)

    def get_params(self, deep=True):
        return {**self.kwargs}

    def set_params(self, **parameters):
        self.model = self.z_clazz(**parameters)
        return self

    

# @Container.bind()
# class GridSearchCVPlugin(BasePlugin, BaseFilter):
#     def __init__(self, filterx:  Dict[str, Any]): #TODO primero implemento con un solo filtro
#         # super().__init__(filterx)
#         clazz:Type[BaseFilter] = Container.ff.get(filterx['clazz'])
        

#         if clazz is not None:
#             obj = clazz(**filterx)
#             print(obj)
           

#     def fit(self, x, y):
#         self.model.fit(x, y) # type: ignore
    
#     def predict(self, x ):
#         return self.model.predict(x) # type: ignore
    

dummy_filter:BaseFilter = Container.ff.SkFilterWrapper(param1=[1, 2, 3], param2=[0.1, 0.2, 0.3], scoring='f1_weighted') # type: ignore
dummy_grid_search_cv:BaseFilter = GridSearchCVPlugin(filterx=dummy_filter)  # type: ignore # type: ignore)

# print(" Grid Search","*"*100)
# print(grid_search.model_dump())
# print("*"*100)

try:

    DummyFilter(param1="hello", param2=[0.1, 0.2, 0.3]) # type: ignore

except Exception as e:
    print("TODO Ok")
    

# except Exception as e:
#     print("\n"*5)
#     print("Exception occurred:", e)