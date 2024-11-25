from typing import Dict, List, Optional, Type, cast

from pydantic import BaseModel, ConfigDict
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BaseMetric, BasePipeline, BasePlugin
from framework3.container.container import Container
from framework3.container.model.bind_model import BindGenericModel

@Container.bind()
class F3Pipeline(BasePipeline):
    model_config = ConfigDict(extra='allow')
    def __init__(self, plugins: List[BasePlugin], metrics:List[BaseMetric], overwrite:bool = False, store:bool = False, log: bool = False):
        super().__init__()
        self.plugins: List[BasePlugin] = plugins
        self.metrics: List[BaseMetric] = metrics
        self.overwrite = overwrite
        self.store = store
        self.log = log
        self._filters: Dict[str, BaseFilter] = {}

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
        print('Fitting pipeline...')
        for plugin in self.plugins:
            print(plugin.__class__.__name__)
            config = plugin.model_dump()
            binding:Optional[BindGenericModel[BaseFilter]] = Container.ff.get(plugin.__class__.__name__)

            if binding is None:
                Container.ff.print_available_filters()
                raise ValueError(f'Filter {config["clazz"]} not found')
            
            filter_:BaseFilter = binding.filter(**config)

            filter_.fit(x, y)
            x = filter_.predict(x)

            self._filters[plugin.__class__.__name__] = filter_
    
    def predict(self, x:XYData) -> XYData:
        if not self._filters:
            raise ValueError('No filters have been trained yet')
        print('Predicting pipeline...')
        for name, filter_ in self._filters.items():
            print(name)
            x = filter_.predict(x)
            
        return x
    
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Dict[str, float]:
        print('Evaluating pipeline...')
        evaluations = {}
        for metric in self.metrics:
            print(metric.__class__.__name__)
            evaluations[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)

        return evaluations



