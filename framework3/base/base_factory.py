from typing import Dict, Iterator, Tuple, Type, Generic
from framework3.base.base_types import TypePlugable
from rich import print as rprint



class BaseFactory(Generic[TypePlugable]):
    def __init__(self):
        self._foundry: Dict[str, Type[TypePlugable]] = {}

    def __getattr__(self, name: str) -> Type[TypePlugable]:
        if name in self._foundry:
            return self._foundry[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Type[TypePlugable]) -> None:
        if name == '_foundry':
            super().__setattr__(name, value)
        else:
            self._foundry[name] = value

    def __setitem__(self, name: str, value: Type[TypePlugable]) -> None:
        if name == '_foundry':
            super().__setattr__(name, value)
        else:
            self._foundry[name] = value

    def __getitem__(self, name: str, default:Type[TypePlugable]|None=None) -> Type[TypePlugable]:
        if name in self._foundry:
            return self._foundry[name]
        else:
            if default is None:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return default

    def __iter__(self) -> Iterator[Tuple[str, Type[TypePlugable]]]:
        return iter(self._foundry.items())

    def __contains__(self, item: str) -> bool:
        return item in self._foundry

    def get(self, name: str, default:Type[TypePlugable]|None=None) -> Type[TypePlugable]:
        if name in self._foundry:
            return self._foundry[name]
        else:
            if default is None:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return default
    
    def print_available_components(self):
        rprint(f"[bold]Available {self.__class__.__name__[:-7]}s:[/bold]")
        for name, binding in self._foundry.items():
            rprint(f"  - [green]{name}[/green]: {binding}")