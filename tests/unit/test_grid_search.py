from typing import List, Optional
import pytest
import numpy as np
from framework3.container.container import Container
from framework3.plugins.filters.grid_search.cv_grid_search import GridSearchCVPlugin
from framework3.base.base_clases import BasePlugin, BaseFilter
from framework3.base.base_types import XYData, VData

@Container.bind()
class DummyFilter(BaseFilter, BasePlugin):
    def __init__(self, param1: int=0 , param2: float=1.0):
        super().__init__(param1=param1, param2=param2)
        
    
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        pass

    def predict(self, x: XYData) -> XYData:
        return x 
    @staticmethod
    def item_grid(param1: List[int], param2: List[float]) -> dict:
        return {"filterx": DummyFilter, "param_grid": {"param1": param1, "param2": param2}}

def test_grid_search_with_list_params():
    # Crear un DummyFilter con listas de parámetros
    grid_search = GridSearchCVPlugin(scoring='f1_weighted', **DummyFilter.item_grid(param1=[1, 2, 3], param2=[0.1, 0.2, 0.3]))
    
    # Verificar que GridSearchCVPlugin se ha creado correctamente
    assert isinstance(grid_search, GridSearchCVPlugin)
    
    # Verificar que los parámetros se han pasado correctamente
    assert grid_search._clf.get_params().get('param_grid') == {'param1': [1, 2, 3], 'param2': [0.1, 0.2, 0.3]}

def test_grid_search_fit_and_predict():
    grid_search = GridSearchCVPlugin(scoring='f1_weighted', **DummyFilter.item_grid(param1=[1, 2], param2=[0.1, 0.2]))
    
    # Crear datos de prueba
    X = XYData.mock( np.array(list(range(10000))))
    y = XYData.mock( np.array(list(range(10000))))
    
    # Fit
    grid_search.fit(X, y)
    
    # Predict
    predictions = grid_search.predict(X)
    
    assert predictions.value.shape == (10000,)

def test_grid_search_best_params():
    grid_search = GridSearchCVPlugin(scoring='f1_weighted', **DummyFilter.item_grid(param1=[1, 2], param2=[0.1, 0.2]))
    
    X = XYData.mock( np.array(list(range(10000))))
    y = XYData.mock( np.array(list(range(10000))))
    
    grid_search.fit(X, y)
    
    assert hasattr(grid_search._clf, 'best_params_')
    assert isinstance(grid_search._clf.best_params_, dict)
    assert 'param1' in grid_search._clf.best_params_
    assert 'param2' in grid_search._clf.best_params_

@pytest.mark.parametrize("param1,param2", [
    ([1, 2], [0.1]),
    ([1], [0.1, 0.2]),
    ([1, 2], [0.1, 0.2])
])
def test_grid_search_mixed_params(param1, param2):
    grid_search = GridSearchCVPlugin(scoring='f1_weighted', **DummyFilter.item_grid(param1=[1, 2], param2=[0.1, 0.2]))
    
    X = XYData.mock( np.array(list(range(10000))))
    y = XYData.mock( np.array(list(range(10000))))
    
    grid_search.fit(X, y)
    predictions = grid_search.predict(X)
    
    assert predictions.value.shape == (10000,)