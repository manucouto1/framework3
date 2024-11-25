from typing import Dict, Any, Iterator, Tuple, Type, Optional, TypeVar, Generic
from framework3.container.model.bind_model import BindGenericModel
from framework3.base.base_types import TypePlugable
from rich import print as rprint



class BaseFactory(Generic[TypePlugable]):
    def __init__(self):
        self._bindings: Dict[str, BindGenericModel[TypePlugable]] = {}

    def __getattr__(self, name: str) -> Type[TypePlugable]:
        if name in self._bindings:
            return self._bindings[name].filter
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Type[TypePlugable]) -> None:
        if name == '_bindings':
            super().__setattr__(name, value)
        else:
            self._bindings[name] = BindGenericModel[TypePlugable](filter=value)

    def __setitem__(self, name: str, value: BindGenericModel[TypePlugable]) -> None:
        if name == '_bindings':
            super().__setattr__(name, value)
        else:
            self._bindings[name] = value

    def __getitem__(self, name: str, default:BindGenericModel[TypePlugable]|None=None) -> BindGenericModel[TypePlugable]:# -> BindGenericModel[TypePlugable] | Any:
        if name in self._bindings:
            return self._bindings[name]
        else:
            if default is None:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return default

    def __iter__(self) -> Iterator[Tuple[str, BindGenericModel[TypePlugable]]]:
        return iter(self._bindings.items())

    def __contains__(self, item: str) -> bool:
        return item in self._bindings

    def get(self, name: str, default:BindGenericModel[TypePlugable]|None=None) -> BindGenericModel[TypePlugable]:
        if name in self._bindings:
            return self._bindings[name]
        else:
            if default is None:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return default
    
    def print_available_components(self):
        rprint(f"[bold]Available {self.__class__.__name__[:-7]}s:[/bold]")
        for name, binding in self._bindings.items():
            rprint(f"  - [green]{name}[/green]: {binding.filter}")