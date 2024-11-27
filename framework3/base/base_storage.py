from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Any, Type


from framework3.base.base_clases import BasePlugin
from typing import TypeVar, cast, Any

class BaseSingleton:
    _instances: Dict[Type[BaseSingleton], Any] = {}

    def __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls) # type: ignore
        return cls._instances[cls]

class BaseStorage(BasePlugin, BaseSingleton):
    @abstractmethod
    def get_root_path(self) -> str:...
    @abstractmethod
    def upload_file(self, file:object, file_name:str, context:str, direct_stream:bool=False) -> str|None: ...
    @abstractmethod
    def download_file(self, hashcode:str, context:str) -> Any:...
    @abstractmethod
    def list_stored_files(self, context:str) -> List[Any]:...
    @abstractmethod
    def get_file_by_hashcode(self, hashcode:str, context:str) -> Any:...
    def check_if_exists(self, hashcode:str, context:str) -> bool:...
    @abstractmethod
    def delete_file(self, hashcode:str, context:str):...
        