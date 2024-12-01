import pickle
from typing import List, Any, Dict, Sequence
from framework3.base import BasePipeline, XYData, BaseFilter
from framework3.base.base_types import VData
from framework3.map_reduce.pyspark import PySparkMapReduce

class MapReduceFeatureExtractorPipeline(BasePipeline):
    def __init__(self, filters: Sequence[BaseFilter], app_name: str, master: str = "local", numSlices: int = 4):
        super().__init__(filters=filters)
        self.filters = filters
        self.numSlices = numSlices
        self._map_reduce = PySparkMapReduce(app_name, master)

    def init(self):
        # Inicialización del pipeline
        pass

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        try:
            self.fit(x, y)
            return self.predict(x)
        except Exception as e:
            # Manejar la excepción apropiadamente
            raise e

    def fit(self, x: XYData, y: XYData | None = None):
        def fit_function(filter):
            filter.fit(x, y)
            return filter
        
        # Aplicar fit en paralelo a los filtros
        rdd = self._map_reduce.map(self.filters, fit_function, numSlices=self.numSlices)

        # Actualizar los filtros con las versiones entrenadas
        self.filters = rdd.collect()

    def predict(self, x: XYData) -> XYData:
        def predict_function(filter:BaseFilter) -> tuple[str, VData]:
            result:XYData = filter.predict(x)
            m_hash,_= filter._get_model_key(x._hash)
            return m_hash, XYData.ensure_dim(result.value)
        
        # Aplicar predict en paralelo a los filtros
        self._map_reduce.map(self.filters, predict_function, numSlices=self.numSlices)
        aux = self._map_reduce.reduce(lambda x: x)
        # Reducir los resultados
        aux = dict(aux)
        print(aux)
        aux = aux.values()
        print(aux)
        aux = list(aux)
        print(aux)
        return  XYData.concat(aux, axis=-1)

    def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
        # Implementar la evaluación si es necesario
        return {}

    def log_metrics(self):
        # Implementar el registro de métricas si es necesario
        pass

    def finish(self):
        self._map_reduce.stop()