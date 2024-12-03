from abc import ABC, abstractmethod
from typing import Any, Callable


class MapReduceStrategy(ABC):
    @abstractmethod
    def map(self, data: Any, map_function: Callable) -> Any:
        pass

    @abstractmethod
    def reduce(self, reduce_function: Callable) -> Any:
        pass

    @abstractmethod
    def stop(self) -> None: ...
