from __future__ import annotations

from typing import Any, Type, Optional, TypeVar

from framework3.base import BaseDatasetManager, BaseFactory
from framework3.base import BaseFilter, BaseMetric, BasePlugin
from framework3.base import BaseStorage
from framework3.base import BasePipeline
from framework3.container.overload import fundispatch


F = TypeVar("F", bound=type)

__all__ = ["Container"]


class Container:
    """
    A container class for managing various components of the framework.

    This class provides a centralized location for storing and managing different types of
    objects such as filters, pipelines, metrics, storage, and plugins. It uses factories
    to create and store these objects.

    Attributes:
        storage (BaseStorage): An instance of BaseStorage for handling storage operations.
        ff (BaseFactory[BaseFilter]): Factory for creating and storing BaseFilter objects.
        pf (BaseFactory[BasePipeline]): Factory for creating and storing BasePipeline objects.
        mf (BaseFactory[BaseMetric]): Factory for creating and storing BaseMetric objects.
        sf (BaseFactory[BaseStorage]): Factory for creating and storing BaseStorage objects.
        pif (BaseFactory[BasePlugin]): Factory for creating and storing BasePlugin objects.

    Example:
        Create and register components in the Container:
        ```python

        from framework3.container import Container
        from framework3.base import BaseFilter, BasePipeline

        @Container.bind()
        class MyFilter(BaseFilter):
            def fit(self, x, y):
                pass
            def predict(self, x):
                return x

        @Container.bind()
        class MyPipeline(BasePipeline):
            def fit(self, x, y):
                pass
            def predict(self, x):
                return x
            def init(self):
                pass
            def start(self, x, y, X_):
                return None
            def log_metrics(self):
                pass
            def finish(self):
                pass
            def evaluate(self, x_data, y_true, y_pred):
                return {}

        # Retrieving and using registered components
        filter_instance = Container.ff["MyFilter"]()
        pipeline_instance = Container.pf["MyPipeline"]()

        result = pipeline_instance.run(filter_instance.process("hello"))
        print(result)  # Output: Processed: HELLO

        ```
    """

    storage: BaseStorage
    ds: BaseDatasetManager
    ff: BaseFactory[BaseFilter] = BaseFactory[BaseFilter]()
    pf: BaseFactory[BasePipeline] = BaseFactory[BasePipeline]()
    mf: BaseFactory[BaseMetric] = BaseFactory[BaseMetric]()
    sf: BaseFactory[BaseStorage] = BaseFactory[BaseStorage]()
    pif: BaseFactory[BasePlugin] = BaseFactory[BasePlugin]()

    @staticmethod
    def bind(manager: Optional[Any] = dict, wrapper: Optional[Any] = dict):
        """
        A decorator for binding various components to the Container.

        This method uses function dispatching to register different types of components
        (filters, pipelines, metrics, storage) with their respective factories in the Container.

        Args:
            manager (Optional[Any]): An optional manager for the binding process. Defaults to dict.
            wrapper (Optional[Any]): An optional wrapper for the binding process. Defaults to dict.

        Returns:
            (Callable): A decorator function that registers the decorated class with the appropriate factory.

        Raises:
            NotImplementedError: If no decorator is registered for the given function.
        """

        @fundispatch  # type: ignore
        def inner(func: Any):
            raise NotImplementedError(f"No decorator registered for {func.__name__}")

        @inner.register(BaseFilter)  # type: ignore
        def _(func: Type[BaseFilter]) -> Type[BaseFilter]:
            Container.ff[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BasePipeline)  # type: ignore
        def _(func: Type[BasePipeline]) -> Type[BasePipeline]:
            Container.pf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BaseMetric)  # type: ignore
        def _(func: Type[BaseMetric]) -> Type[BaseMetric]:
            Container.mf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BaseStorage)  # type: ignore
        def _(func: Type[BaseStorage]) -> Type[BaseStorage]:
            Container.sf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BasePlugin)  # type: ignore
        def _(func: Type[BasePlugin]) -> Type[BasePlugin]:
            Container.pif[func.__name__] = func
            return func

        return inner
