# from __future__ import annotations
# from typing import Callable, ClassVar, List, Literal, Tuple, Type, TypeVar, Union, cast, get_type_hints, Dict, Optional, Any

# from fastapi.encoders import jsonable_encoder
# from framework3.base.base_types import Float
# from typeguard import typechecked
# from pydantic import ConfigDict, create_model, Field, BaseModel
# from abc import ABC, abstractmethod
# from framework3.base.base_types import XYData, PyDanticMeta, VData
# from framework3.base.base_factory import BaseFactory
# import inspect
# import numpy as np
# import hashlib
# from rich import print as rprint

# T = TypeVar('T')


# class PydanticWrapper(BaseModel):
#     model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)


# class BasePlugin(ABC):

#     def __init__(self, **kwargs):
#         self._pydantic_model = PydanticWrapper(**kwargs)

#     def __getattr__(self, name):
#         try:
#             return getattr(self._pydantic_model, name)
#         except AttributeError:
#             raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self._pydantic_model})"

#     def model_dump(self, **kwargs):
#         return self._pydantic_model.model_dump(**kwargs)

#     def dict(self, **kwargs):
#         return self._pydantic_model.model_dump(**kwargs)

#     def json(self, **kwargs):
#         return self._pydantic_model.model_dump_json(**kwargs)

#     def item_dump(self, **kwargs) -> Dict[str, Any]:
#         print(f"Dumping {self.__class__.__name__}")
#         return {
#             'clazz': self.__class__.__name__,
#             'params': jsonable_encoder(self._pydantic_model.__pydantic_extra__,
#                     # exclude_none=True,
#                     custom_encoder={
#                         BasePlugin: lambda v: v.item_dump(),
#                         Type: lambda v: {'clazz': v.__name__}
#                     },
#                     **kwargs
#                 )
#             }

#     @classmethod
#     def model_validate(cls, obj):
#         pydantic_obj = PydanticWrapper.model_validate(obj)
#         return cls(**pydantic_obj.model_dump())

#     def __rich_repr__(self):
#         for key, value in self._pydantic_model:
#             yield key, value


#     @staticmethod
#     def build_from_dump(dump_dict:Dict[str, Any], factory:BaseFactory[BasePlugin]) -> BasePlugin|Type[BasePlugin]:
#         level_clazz:Type[BasePlugin] = factory[dump_dict['clazz']]
#         print(f"Building {dump_dict['clazz']}")
#         if 'params' in dump_dict:
#             level_params={}
#             for k,v in dump_dict['params'].items():
#                 if isinstance(v, dict):
#                     if 'clazz' in v:
#                         level_params[k]=BasePlugin.build_from_dump(v, factory)
#                     else:
#                         level_params[k]=v
#                 elif isinstance(v, list):
#                     level_params[k] = [BasePlugin.build_from_dump(i, factory) for i in v]
#                 else:
#                     level_params[k]=v
#             return level_clazz(**level_params)
#         else:
#             return level_clazz


#     @staticmethod
#     def item_grid(*args, **kwargs) -> Dict[str, Any]: ...


# class MetaBasePlugin(PyDanticMeta):
#     def __new__(cls, name, bases, attrs):

#         new_class = super().__new__(cls, name, bases, attrs)

#         init = attrs.get("__init__")
#         if init:  # Solo si se define __init__ en la clase actual
#             new_class = cls._add_init_args_to_model_fields(new_class, init)  # Agregar argumentos de __init__ como campos Pydantic
#             new_class.__init__ = typechecked(init)  # Aplicar comprobación de tipos al constructor


#         new_class = cls._inherit_annotations(new_class, bases)
#         new_class = cls._type_check_inherit_methods(new_class)

#         return new_class

#     @staticmethod
#     def _add_init_args_to_model_fields(new_cls, init):
#         sig = inspect.signature(init)
#         annotations = get_type_hints(init)

#         # Agregar argumentos de __init__ como campos Pydantic
#         for param_name, param in sig.parameters.items():
#             if param_name not in ("self", "kwargs"):
#                 if param_name not in new_cls.__annotations__:
#                     # Anotar dinámicamente los campos
#                     new_cls.__annotations__[param_name] = annotations.get(param_name, param.annotation)

#         return new_cls

#     @staticmethod
#     def _type_check_inherit_methods(new_cls):
#         # Aplicar typechecked a métodos concretos
#         for attr_name, attr_value in new_cls.__dict__.items():
#             if inspect.isfunction(attr_value) and \
#                 not getattr(attr_value, '__isabstractmethod__', False) and attr_name != '__init__':
#                 original_func = attr_value
#                 wrapped_func = typechecked(attr_value)
#                 wrapped_func.__wrapped__ = original_func  # type: ignore # Referencia al método original
#                 setattr(new_cls, attr_name, wrapped_func)
#         return new_cls
#     @staticmethod
#     def _inherit_annotations(ccls, bases):
#         """Heredar anotaciones de tipo de métodos abstractos."""
#         for base in bases:
#             for attr_name, attr_value in base.__dict__.items():
#                 if getattr(attr_value, '__isabstractmethod__', False):
#                     abstract_annotations = get_type_hints(attr_value)
#                     if hasattr(ccls, attr_name):
#                         concrete_method = getattr(ccls, attr_name)
#                         if callable(concrete_method):
#                             combined_annotations = {**abstract_annotations, **get_type_hints(concrete_method)}
#                             #combined_annotations = {**abstract_annotations}
#                             concrete_method.__annotations__ = combined_annotations
#                             setattr(ccls, attr_name, concrete_method)

#         return ccls

# class BasePlugin( ABC, BaseModel, metaclass=MetaBasePlugin):
#     model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
#     def item_dump(self, **kwargs) -> dict[str, Any]:
#         return {
#             'clazz': self.__class__.__name__,
#             'params': jsonable_encoder(self.__pydantic_extra__, custom_encoder={BasePlugin: lambda v: v.item_dump(), Type: lambda v: {'clazz': v.__name__}}, **kwargs),
#             # 'extra_params': {k: v for k, v in self.__dict__.items() if k.startswith("_")}
#         }

#     @staticmethod
#     def build_from_dump(dump_dict:Dict[str, Any], factory:BaseFactory[BasePlugin]) -> BasePlugin|Type[BasePlugin]:
#         level_clazz:Type[BasePlugin] = factory[dump_dict['clazz']]
#         print(f"Building {dump_dict['clazz']}")
#         if 'params' in dump_dict:
#             level_params={}
#             for k,v in dump_dict['params'].items():
#                 if isinstance(v, dict):
#                     if 'clazz' in v:
#                         level_params[k]=BasePlugin.build_from_dump(v, factory)
#                     else:
#                         level_params[k]=v
#                 elif isinstance(v, list):
#                     level_params[k] = [BasePlugin.build_from_dump(i, factory) for i in v]
#                 else:
#                     level_params[k]=v
#             return level_clazz(**level_params)
#         else:
#             return level_clazz
