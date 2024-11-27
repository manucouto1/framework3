from typing import Dict, List, Optional, Tuple, Type, cast

from pydantic import ConfigDict
from framework3.base.base_types import XYData, VData
from framework3.base.base_clases import BaseFilter, BaseMetric, BasePipeline, BasePlugin
from framework3.container.container import Container
from framework3.container.model.bind_model import BindGenericModel

from rich import print as rprint

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
        self._filters: Dict[str, BaseFilter] = {}
        self._filters_str: Dict[str, str] = {}


    def init(self): ... #TODO Este método inicializará el logger, seguramente wandb

    def start(self, x:XYData, y:Optional[XYData], X_:Optional[XYData]) -> Optional[VData]:
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
        level_path = x._path
        
        if self._filters:
            self._filters = {}
            self._filters_str = {}

        for plugin in self.plugins:
            filter_: BaseFilter = cast(BaseFilter, plugin)
            
            # Congiguración del filtro
            m_hash, m_str = filter_._get_model_key(data_hash=f'{x._hash}, {y._hash if y is not None else ""}')
            m_path = f'{level_path}/{m_hash}'

            rprint(f'{m_str}')
            # Datos del modelo
            x = XYData(_hash=m_hash,_path=m_path,_value=x.value)
            rprint(f'\t - Model: {x}')
            filter_.fit(x, y)
            
            # Configuración de los datos generados por el filtro
            d_hash, d_str = filter_._get_data_key(m_str, x._hash)
            # Datos de entrenamiento
            x = XYData(_hash=d_hash,_path=m_path,_value=x._value)
            rprint(f'{d_str}')
            rprint(f'\t - Data: {x}')
            value = filter_.predict(x)
            
            # Paso de datos al siguiente filtro
            x = XYData(_hash=d_hash,_path=m_path,_value=value)
            # Filtro entrenado
            self._filters[m_hash] = filter_
            self._filters_str[m_hash] = m_str
    
    def predict(self, x:XYData) -> VData:
        if not self._filters:
            raise ValueError('No filters have been trained yet')
        rprint('_'*100)
        rprint('Predicting pipeline...')
        rprint('*'*100)
        level_path = x._path
        for m_hash, filter_ in self._filters.items():
            m_str = self._filters_str[m_hash]
            d_hash, d_str = filter_._get_data_key(m_str, x._hash)
            m_path = f'{level_path}/{m_hash}'

            x = XYData(_hash=d_hash,_path=m_path,_value=x._value)
            rprint(f'{d_str}')
            rprint(f'\t - Data: {x}')
            value = filter_.predict(x)
            x = XYData(_hash=d_hash,_path=m_path,_value=value)
            
        return x.value
    
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Dict[str, float]:
        rprint('Evaluating pipeline...')
        evaluations = {}
        for metric in self.metrics:
            evaluations[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        return evaluations



