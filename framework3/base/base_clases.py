from typing import ClassVar, List, Literal, Tuple, Union, cast, get_type_hints, Dict, Optional, Any
from framework3.base.base_types import Float
from typeguard import typechecked
from pydantic import ConfigDict, create_model, Field, BaseModel
from abc import ABC, abstractmethod
from framework3.base.base_types import XYData, PyDanticMeta, IncEx
import inspect
import numpy as np
from rich import print

class MetaBasePlugin(PyDanticMeta):
    def __new__(cls, name, bases, attrs):

        new_class = super().__new__(cls, name, bases, attrs)

        init = attrs.get("__init__")
        if init:  # Solo si se define __init__ en la clase actual
            new_class.__init__ = typechecked(init)  # Aplicar comprobación de tipos al constructor

        def custom_item_dump(self, **kwargs):
            return {
                'clazz': self.__class__.__name__,
                'params': self.model_dump(**kwargs),
                'extra_params': {k: v for k, v in self.__dict__.items() if k.startswith("_")}
            }

        setattr(new_class, 'item_dump', custom_item_dump)
        new_class = cls._inherit_annotations(new_class, bases)
        new_class = cls._type_check_inherit_methods(new_class)
        return new_class

        
    @staticmethod
    def _type_check_inherit_methods(new_cls):
        # Aplicar typechecked a métodos concretos
        for attr_name, attr_value in new_cls.__dict__.items():
            if inspect.isfunction(attr_value) and \
                not getattr(attr_value, '__isabstractmethod__', False) and attr_name != '__init__':
                original_func = attr_value
                wrapped_func = typechecked(attr_value)
                wrapped_func.__wrapped__ = original_func  # type: ignore # Referencia al método original
                setattr(new_cls, attr_name, wrapped_func)
        return new_cls
    @staticmethod
    def _inherit_annotations(ccls, bases):
        """Heredar anotaciones de tipo de métodos abstractos."""
        for base in bases:
            for attr_name, attr_value in base.__dict__.items():
                if getattr(attr_value, '__isabstractmethod__', False):
                    abstract_annotations = get_type_hints(attr_value)
                    if hasattr(ccls, attr_name):
                        concrete_method = getattr(ccls, attr_name)
                        if callable(concrete_method):
                            combined_annotations = {**abstract_annotations, **get_type_hints(concrete_method)}
                            #combined_annotations = {**abstract_annotations}
                            concrete_method.__annotations__ = combined_annotations
                            setattr(ccls, attr_name, concrete_method)

        return ccls
    
    
class BasePlugin( ABC, BaseModel, metaclass=MetaBasePlugin):
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    
    def item_dump(
        self,
        *,
        mode: Literal['json', 'python'] | str = 'python',
        include: IncEx = None,
        exclude: IncEx = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
    ) -> dict[str, Any]: ...

    @staticmethod
    def item_grid(*args, **kwargs) -> Dict[str, Any]: ...

class BaseFilter(BasePlugin):
    @abstractmethod
    def fit(self, x:XYData, y:Optional[XYData]) -> None: ...
    @abstractmethod
    def predict(self, x:XYData) -> XYData:...

class BasePipeline(BaseFilter):
    @abstractmethod
    def init(self) -> None: ...
    @abstractmethod
    def start(self, x:XYData, y:Optional[XYData], X_:Optional[XYData]) -> Optional[XYData]:...
    @abstractmethod
    def log_metrics(self) -> None: ...
    @abstractmethod
    def finish(self) -> None: ...
    @abstractmethod
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Dict[str, float]: ...

class BaseMetric(BasePlugin):
    @abstractmethod
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Float|np.ndarray: ...

class BaseStorage(BasePlugin):
    @abstractmethod
    def upload_file(self, file:object, file_name:str, direct_stream:bool=False) -> str|None: ...
    @abstractmethod
    def download_file(self, hashcode:str) -> Any:...
    @abstractmethod
    def list_stored_files(self) -> List[Any]:...
    @abstractmethod
    def get_file_by_hashcode(self, hashcode:str) -> Any:...
    @abstractmethod
    def check_if_exists(self, hashcode:str) -> bool:...
    @abstractmethod
    def delete_file(self, hashcode:str):...
        