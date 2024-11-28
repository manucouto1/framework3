from typing import Callable, Dict, Optional, Tuple, cast
from framework3.container.container import Container
from framework3.base.base_clases import BaseFilter
from framework3.base.base_storage import BaseStorage
from framework3.base.base_types import XYData, VData

import functools
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

    def _pre_fit_wrapp(self, method):
        @functools.wraps(method)
        def wrapper(x:XYData, y:Optional[XYData], *args, **kwargs):
            m_hash, m_str = self._get_model_key(data_hash=f'{x._hash}, {y._hash if y is not None else ""}')
            m_path = f'{self._get_model_name()}/{m_hash}'

            rprint(f'\n\t - {m_str}\n')
            
            setattr(self.filter, '_m_hash', m_hash)
            setattr(self.filter, '_m_path', m_path)
            setattr(self.filter, '_m_str', m_str)
            return method(x, y, *args, **kwargs)
        return wrapper
    
    def _pre_predict_wrapp(self, method):
        @functools.wraps(method)
        def wrapper(x:XYData, *args, **kwargs) -> XYData:
            if not self.filter._m_hash or not self.filter._m_path or not self.filter._m_str:
                raise ValueError("Cached filter model not trained or loaded")
            
            d_hash, d_str = self._get_data_key(self.filter._m_str, x._hash)
            rprint(f'\n\t - {d_str}\n')
            new_x = XYData(
                _hash=d_hash,
                _value=x._value,
                _path=f'{self._get_model_name()}/{self.filter._m_hash}',
            )

            return method(new_x, *args, **kwargs)
        return wrapper

    def _get_model_name(self) -> str:
        return self.filter._get_model_name()

    def _get_model_key(self, data_hash:str) -> Tuple[str, str] :
        return BaseFilter._get_model_key(self.filter, data_hash)

    def _get_data_key(self, model_str:str, data_hash:str) -> Tuple[str, str]:
        return BaseFilter._get_data_key(self.filter, model_str, data_hash)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        if not self._storage.check_if_exists('model', f'{self._storage.get_root_path()}/{self.filter._m_path}') or self.overwrite:
            rprint(f"\t - El filtro {self.filter} con hash {self.filter._m_hash} No existe, se va a entrenar.")
            self.filter.fit(x, y)
            if self.cache_filter:
                rprint(f"\t - El filtro {self.filter} Se cachea.")
                self._storage.upload_file(pickle.dumps(self.filter), 'model', context=f'{self._storage.get_root_path()}/{self.filter._m_path}')
        else:
            rprint(f"\t - El filtro {self.filter} Existe, se crea lambda.")
            self._lambda_filter = lambda: cast(BaseFilter, self._storage.download_file('model', f'{self._storage.get_root_path()}/{self.filter._m_path}'))

        

    def predict(self, x: XYData) -> XYData:
        if not self._storage.check_if_exists(x._hash, context=f'{self._storage.get_root_path()}/{x._path}') or self.overwrite:
            rprint(f"\t - El dato {x} No existe, se va a crear.")
            if self._lambda_filter is not None:
                rprint(f"\t - Existe un Lambda por lo que se recupera el filtro del storage.")
                self.filter = self._lambda_filter()
                print(self.filter)

            value = self.filter.predict(x)

            if self.cache_data:
                rprint(f"\t - El dato {x} Se cachea.")
                self._storage.upload_file(pickle.dumps(value.value), x._hash, context=f'{self._storage.get_root_path()}/{x._path}')
        else:
            rprint(f"\t - El dato {x} Existe, se crea lambda.")
            value = XYData(
                _hash=x._hash,
                _path=x._path,
                _value=lambda: cast(VData, self._storage.download_file(x._hash, f'{self._storage.get_root_path()}/{x._path}'))
            )
        return value

    def clear_cache(self):
        # Implementa la lógica para limpiar el caché en el almacenamiento
        pass