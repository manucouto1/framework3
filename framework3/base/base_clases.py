from __future__ import annotations
from typing import Callable, ClassVar, List, Literal, Tuple, Type, TypeVar, Union, cast, get_type_hints, Dict, Optional, Any

from fastapi.encoders import jsonable_encoder
import typeguard
from framework3.base.base_types import Float
from typeguard import typechecked
from pydantic import ConfigDict, create_model, Field, BaseModel
from abc import ABC, abstractmethod
from framework3.base.base_types import XYData, PyDanticMeta, VData
from framework3.base.base_factory import BaseFactory
import inspect
import numpy as np
import hashlib
from rich import print as rprint

T = TypeVar('T')


class BasePlugin(ABC):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        # Obtener el método __init__ de la clase actual
        init_method = cls.__init__
        # Aplicar typechecked al método __init__
        if init_method is not object.__init__:
            cls.__init__ = typechecked(init_method)

        # Heredar anotaciones de tipo de métodos abstractos
        cls.__inherit_annotations()

        # Aplicar typechecked a todos los métodos definidos en la clase
        for attr_name, attr_value in cls.__dict__.items():
            if inspect.isfunction(attr_value) and attr_name != '__init__':
                setattr(cls, attr_name, typechecked(attr_value))


        return instance
    
    @classmethod
    def __inherit_annotations(cls):
        for base in cls.__bases__:
            for name, method in base.__dict__.items():
                if getattr(method, '__isabstractmethod__', False):
                    if hasattr(cls, name):
                        concrete_method = getattr(cls, name)
                        abstract_annotations = get_type_hints(method)
                        concrete_annotations = get_type_hints(concrete_method)
                        combined_annotations = {**abstract_annotations, **concrete_annotations}
                        setattr(concrete_method, '__annotations__', combined_annotations)


    def __init__(self, **kwargs):
        self.__dict__['_public_attributes'] = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        self.__dict__['_private_attributes'] = {k: v for k, v in kwargs.items() if k.startswith("_")}

    def __getattr__(self, name):
        if name in self._public_attributes:
            return self._public_attributes[name]
        elif name in self._private_attributes:
            return self._private_attributes[name]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            self._private_attributes[name] = value
        else:
            self._public_attributes[name] = value

    def __getitem__(self, key: str) -> Any:
        if key in self._public_attributes:
            return self._public_attributes[key]
        elif key in self._private_attributes:
            return self._private_attributes[key]
        raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")


    def __repr__(self):
        return f"{self.__class__.__name__}({self._public_attributes})"

    def model_dump(self, **kwargs):
        return self._public_attributes.copy()

    def dict(self, **kwargs):
        return self.model_dump(**kwargs)

    def json(self, **kwargs):
        return jsonable_encoder(self._public_attributes, **kwargs)

    def item_dump(self, **kwargs) -> Dict[str, Any]:
        print(f"Dumping {self.__class__.__name__}")
        return {
            'clazz': self.__class__.__name__,
            'params': jsonable_encoder(
                self._public_attributes,
                custom_encoder={
                    BasePlugin: lambda v: v.item_dump(),
                    type: lambda v: {'clazz': v.__name__}
                },
                **kwargs
            )
        }
    
    def get_extra(self)->Dict[str, Any]:
        return self._private_attributes.copy()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict):
            return cls(**obj)
        raise ValueError(f"Cannot validate {type(obj)}")

    def __rich_repr__(self):
        for key, value in self._public_attributes.items():
            yield key, value

    @staticmethod
    def build_from_dump(dump_dict:Dict[str, Any], factory:BaseFactory[BasePlugin]) -> BasePlugin|Type[BasePlugin]:
        level_clazz:Type[BasePlugin] = factory[dump_dict['clazz']]
        print(f"Building {dump_dict['clazz']}")
        if 'params' in dump_dict:
            level_params={}
            for k,v in dump_dict['params'].items():
                if isinstance(v, dict):
                    if 'clazz' in v:
                        level_params[k]=BasePlugin.build_from_dump(v, factory)
                    else:
                        level_params[k]=v
                elif isinstance(v, list):
                    level_params[k] = [BasePlugin.build_from_dump(i, factory) for i in v]
                else:
                    level_params[k]=v
            return level_clazz(**level_params)
        else:
            return level_clazz

        
class BaseFilter(BasePlugin):
    
    @abstractmethod
    def fit(self, x:XYData, y:Optional[XYData]) -> None: ...

    @abstractmethod
    def predict(self, x:XYData) -> VData|Callable[...,VData]:...

    def _get_model_key(self, data_hash:str) -> Tuple[str, str] :
        model_str = f"<{self.item_dump(exclude='extra_params')}>({data_hash})"
        model_hashcode = hashlib.sha1(model_str.encode('utf-8')).hexdigest()
        return model_hashcode, model_str
    
    def _get_data_key(self, model_str:str, data_hash:str) -> Tuple[str, str]:
        data_str = f"{model_str}.predict({data_hash})"
        data_hashcode = hashlib.sha1(data_str.encode('utf-8')).hexdigest()
        return data_hashcode, data_str

class BasePipeline(BaseFilter):
    @abstractmethod
    def init(self) -> None: ...
    @abstractmethod
    def start(self, x:XYData, y:Optional[XYData], X_:Optional[XYData]) -> Optional[VData]:...
    @abstractmethod
    def log_metrics(self) -> None: ...
    @abstractmethod
    def finish(self) -> None: ...
    @abstractmethod
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Dict[str, float]: ...

class BaseMetric(BasePlugin):
    @abstractmethod
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData) -> Float|np.ndarray: ...
