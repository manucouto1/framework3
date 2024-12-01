from abc import ABC, abstractmethod
from typing import Any, Callable
from framework3.base.base_types import XYData


class MapReduceStrategy(ABC):
    @abstractmethod
    def map(self, data: Any, map_function:Callable) -> Any:
        pass

    @abstractmethod
    def reduce(self, reduce_function:Callable) -> Any:
        pass
    
    @abstractmethod
    def stop(self) -> None: ...