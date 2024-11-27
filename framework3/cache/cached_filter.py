from typing import Callable, Dict, Optional, Tuple, cast
from framework3.container.container import Container
from framework3.base.base_clases import BaseFilter
from framework3.base.base_storage import BaseStorage
from framework3.base.base_types import XYData, VData

from rich import print as rprint
import pickle

@Container.bind()
class Cached(BaseFilter):
    def __init__(self, filter: BaseFilter, cache_data: bool = False, cache_filter: bool = False, overwrite: bool=False, storage:BaseStorage|None=None):
        super().__init__(filter=filter, cache_data=cache_data, cache_filter=cache_filter, overwrite=overwrite, storage=storage)
        self.filter: BaseFilter = filter
        self.cache_data = cache_data
        self.cache_filter = cache_filter
        self.overwrite = overwrite
        self._storage: BaseStorage = Container.storage if storage is None else storage
        self._lambda_filter: Callable[...,BaseFilter]|None = None

    # def _get_model_path(self, base_path:str) -> str:
    #     return BaseFilter._get_model_path(self.filter, base_path)

    def _get_model_key(self, data_hash:str) -> Tuple[str, str] :
        return BaseFilter._get_model_key(self.filter, data_hash)

    def _get_data_key(self, model_str:str, data_hash:str) -> Tuple[str, str]:
        return BaseFilter._get_data_key(self.filter, model_str, data_hash)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        if not self._storage.check_if_exists(x._hash, x._path) or self.overwrite:
            rprint(f"\t - El filtro {self.filter.__class__.__name__} No existe, se va a entrenar.")
            self.filter.fit(x, y)
            if self.cache_filter:
                rprint(f"\t - El filtro {self.filter.__class__.__name__} Se cachea.")
                self._storage.upload_file(pickle.dumps(self.filter), x._hash, context=x._path)
        else:
            rprint(f"\t - El filtro {self.filter.__class__.__name__} Existe, se crea lambda.")
            self._lambda_filter = lambda: cast(BaseFilter, self._storage.download_file(x._hash, x._path))

    def predict(self, x: XYData) -> VData|Callable[...,VData]:
        if not self._storage.check_if_exists(x._hash, x._path) or self.overwrite:
            rprint(f"\t - El dato {x._hash} No existe, se va a crear.")
            if self._lambda_filter is not None:
                rprint(f"\t - Existe un Lambda por lo que se recupera el filtro del storage.")
                self.filter = self._lambda_filter()
                print(self.filter)

            value = self.filter.predict(x)

            if self.cache_data:
                rprint(f"\t - El dato {x._hash} Se cachea.")
                self._storage.upload_file(pickle.dumps(value), x._hash, context=x._path)
        else:
            rprint(f"\t - El dato {x._hash} Existe, se crea lambda.")
            value = lambda: cast(VData, self._storage.download_file(x._hash, x._path))
        
        return value

    def clear_cache(self):
        # Implementa la lógica para limpiar el caché en el almacenamiento
        pass