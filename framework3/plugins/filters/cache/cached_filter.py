from typing import Callable, Dict, Optional, Tuple, cast
from framework3.base.base_clases import BasePipeline
from framework3.container.container import Container
from framework3.base import BaseFilter
from framework3.base import BaseStorage
from framework3.base import XYData, VData

import functools
from rich import print as rprint
import pickle

__all__ = ['Cached']

@Container.bind()
class Cached(BaseFilter):
    """
    A filter that manages the storage of models and data in a BaseStorage type.

    This class extends BaseFilter to provide caching capabilities for both the filter model
    and the processed data. It allows for efficient reuse of previously computed results
    and trained models.

    Args:
        filter (BaseFilter): The underlying filter to be cached.
        cache_data (bool): Whether to cache the processed data.
        cache_filter (bool): Whether to cache the trained filter.
        overwrite (bool): Whether to overwrite existing cached data/models.
        storage (BaseStorage|None): The storage backend for caching.

    Example:
        Ejemplo de uso de la clase:
        ```python

        from framework3.storage import LocalStorage
        from framework3.container import Container
        from your_custom_filter import CustomFilter
             # Configurar el almacenamiento
        Container.storage = LocalStorage(storage_path='cache')
        
        # Crear un filtro personalizado y envolverlo con Cached
        custom_filter = CustomFilter()
        cached_filter = Cached(
            filter=custom_filter,
            cache_data=True,
            cache_filter=True,
            overwrite=False
        )
             # Usar el filtro cacheado
        X = XYData(_hash='input_data', _path='/datasets', _value=input_data)
        y = XYData(_hash='target_data', _path='/datasets', _value=target_data)
             cached_filter.fit(X, y)
        predictions = cached_filter.predict(X)
             # Limpiar el caché si es necesario
        cached_filter.clear_cache()
        ```
    """
    def __init__(self, filter: BaseFilter, cache_data: bool = True, cache_filter: bool = True, overwrite: bool=False, storage:BaseStorage|None=None):
        super().__init__(filter=filter, cache_data=cache_data, cache_filter=cache_filter, overwrite=overwrite, storage=storage)
        self.filter: BaseFilter = filter
        self.cache_data = cache_data
        self.cache_filter = cache_filter
        self.overwrite = overwrite
        self._storage: BaseStorage = Container.storage if storage is None else storage
        self._lambda_filter: Callable[...,BaseFilter]|None = None

    def _pre_fit_wrapp(self, x: XYData, y: Optional[XYData]):
        m_hash, m_str = self._get_model_key(data_hash=f'{x._hash}, {y._hash if y is not None else ""}')
        m_path = f'{self._get_model_name()}/{m_hash}'

        print("*"*100)
        print("SE LLAMA PREFIT")
        print("*"*100)
        
        setattr(self.filter, '_m_hash', m_hash)
        setattr(self.filter, '_m_path', m_path)
        setattr(self.filter, '_m_str', m_str)
        return self._original_fit(x, y)

    def _pre_predict_wrapp(self, x: XYData, *args, **kwargs) -> XYData:
        if not self.filter._m_hash or not self.filter._m_path or not self.filter._m_str:
            raise ValueError("Cached filter model not trained or loaded")
        
        d_hash, d_str = self._get_data_key(self.filter._m_str, x._hash)
        
        new_x = XYData(
            _hash=d_hash,
            _value=x._value,
            _path=f'{self._get_model_name()}/{self.filter._m_hash}',
        )
        return self._original_predict(new_x, *args, **kwargs)
        
    # def _pre_fit_wrapp(self, method):
    #     """
    #     Decorador para el método fit que prepara el hash y la ruta del modelo antes de ajustar.

    #     Args:
    #         method (Callable): El método fit a envolver.

    #     Returns:
    #         Callable: El método fit envuelto.
    #     """
    #     @functools.wraps(method)
    #     def wrapper(x:XYData, y:Optional[XYData], *args, **kwargs):
    #         m_hash, m_str = self._get_model_key(data_hash=f'{x._hash}, {y._hash if y is not None else ""}')
    #         m_path = f'{self._get_model_name()}/{m_hash}'

    #         setattr(self.filter, '_m_hash', m_hash)
    #         setattr(self.filter, '_m_path', m_path)
    #         setattr(self.filter, '_m_str', m_str)
    #         return method(x, y, *args, **kwargs)
    #     return wrapper
    
    # def _pre_predict_wrapp(self, method):
    #     """
    #     Decorador para el método predict que prepara los datos antes de la predicción.

    #     Args:
    #         method (Callable): El método predict a envolver.

    #     Returns:
    #         Callable: El método predict envuelto.
    #     """
    #     @functools.wraps(method)
    #     def wrapper(x:XYData, *args, **kwargs) -> XYData:
    #         if not self.filter._m_hash or not self.filter._m_path or not self.filter._m_str:
    #             raise ValueError("Cached filter model not trained or loaded")
            
    #         d_hash, d_str = self._get_data_key(self.filter._m_str, x._hash)
            
    #         new_x = XYData(
    #             _hash=d_hash,
    #             _value=x._value,
    #             _path=f'{self._get_model_name()}/{self.filter._m_hash}',
    #         )

    #         return method(new_x, *args, **kwargs)
    #     return wrapper

    def _get_model_name(self) -> str:
        """
        Obtiene el nombre del modelo del filtro subyacente.

        Returns:
            str: El nombre del modelo.
        """
        return self.filter._get_model_name()

    def _get_model_key(self, data_hash:str) -> Tuple[str, str] :
        """
        Genera la clave del modelo basada en el hash de los datos.

        Args:
            data_hash (str): El hash de los datos de entrada.

        Returns:
            Tuple[str, str]: Una tupla con el hash del modelo y su representación en string.
        """
        return BaseFilter._get_model_key(self.filter, data_hash)

    def _get_data_key(self, model_str:str, data_hash:str) -> Tuple[str, str]:
        """
        Genera la clave de los datos basada en el modelo y el hash de los datos.

        Args:
            model_str (str): La representación en string del modelo.
            data_hash (str): El hash de los datos de entrada.

        Returns:
            Tuple[str, str]: Una tupla con el hash de los datos y su representación en string.
        """
        return BaseFilter._get_data_key(self.filter, model_str, data_hash)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Ajusta el filtro a los datos de entrada, cacheando el modelo si es necesario.

        Args:
            x (XYData): Los datos de entrada.
            y (Optional[XYData]): Los datos objetivo, si existen.
        """
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
        """
        Realiza predicciones usando el filtro, cacheando los resultados si es necesario.

        Args:
            x (XYData): Los datos de entrada para la predicción.

        Returns:
            XYData: Los resultados de la predicción.
        """
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
        """
        Limpia el caché en el almacenamiento.
        
        Nota: Esta función aún no está implementada.
        """
        # Implementa la lógica para limpiar el caché en el almacenamiento
        pass