import functools
from typing import Callable, Dict, List, Optional, Tuple, Type, cast

from pydantic import ConfigDict
from framework3.base.base_types import XYData, VData
from framework3.base.base_clases import BaseFilter, BaseMetric, BasePipeline, BasePlugin
from framework3.container.container import Container
from framework3.container.model.bind_model import BindGenericModel

from rich import print as rprint
from framework3.utils.utils import dict_to_dataclass

@Container.bind()
class F3Pipeline(BasePipeline):
    model_config = ConfigDict(extra='allow')
    def __init__(self, plugins: List[BasePlugin], metrics:List[BaseMetric], overwrite:bool = False, store:bool = False, log: bool = False) -> None:
        super().__init__(plugins=plugins, metrics=metrics, overwrite=overwrite, store=store, log=log)
        self.plugins: List[BasePlugin] = plugins
        self.metrics: List[BaseMetric] = metrics
        self.overwrite = overwrite
        self.store = store
        self.log = log
        self._filters: List[ BaseFilter] = []

    def _pre_fit_wrapp(self, method):
        @functools.wraps(method)
        def wrapper(x:XYData, y:Optional[XYData], *args, **kwargs):
            return method(x, y, *args, **kwargs)
        return wrapper
    
    def _pre_predict_wrapp(self, method):
        @functools.wraps(method)
        def wrapper(x:XYData, *args, **kwargs) -> XYData:
            return method(x, *args, **kwargs)
        return wrapper


    def init(self): ... #TODO Este método inicializará el logger, seguramente wandb

    def start(self, x:XYData, y:Optional[XYData], X_:Optional[XYData]) -> Optional[XYData]:
        try:
            self.fit(x, y)
            if X_ is not None:
                return self.predict(X_)
            else:
                return self.predict(x)
        except Exception as e:
            print(f'Error during pipeline execution: {e}')
            raise e

    def log_metrics(self):... #TODO en caso de que haya métricas se mostrarán los resultados y se enviarán al logger

    def finish(self):... #TODO Este método finalizará el logger, seguramente wandb
        

    def fit(self, x:XYData, y:Optional[XYData]):
        rprint('_'*100)
        rprint('Fitting pipeline...')
        rprint('*'*100)
        if self._filters:
            self._filters = []

        for plugin in self.plugins:
            filter_: BaseFilter = cast(BaseFilter, plugin)
            rprint(f'\n* {filter_}:')
            filter_.fit(x, y)
            x = filter_.predict(x)
            self._filters.append(filter_)
    
    def predict(self, x:XYData) -> XYData:
        if not self._filters: raise ValueError('No filters have been trained yet')
        rprint('_'*100)
        rprint('Predicting pipeline...')
        rprint('*'*100)

        for filter_ in self._filters:
            rprint(f'\n* {filter_}')
            x = filter_.predict(x)
            
        return x
    
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Dict[str, float]:
        rprint('Evaluating pipeline...')
        evaluations = {}
        for metric in self.metrics:
            evaluations[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        return evaluations



