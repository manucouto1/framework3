from functools import singledispatch, wraps
import os

from pymongo import MongoClient
from typing import Any, Callable, Type, Optional, TypeVar, cast
from dotenv import load_dotenv
from functools import singledispatch

from framework3.base.base_factory import BaseFactory
from framework3.container.factories.storage_factory import StorageFactory
from framework3.container.factories.filter_factory import FilterFactory
from framework3.container.factories.metric_factory import MetricFactory
from framework3.container.factories.pipeline_factory import PipelineFactory
from framework3.container.model.bind_model import BindGenericModel
from framework3.base.base_clases import BaseFilter, BaseMetric, BasePipeline, BasePlugin, BaseStorage
from framework3.utils.method_overload import fundispatch

load_dotenv()

F = TypeVar('F', bound=type)

class Container:
    client: MongoClient = MongoClient(os.environ["MONGO_HOST"], tls=False)
    ff: BaseFactory[BaseFilter] = FilterFactory()
    pf: BaseFactory[BasePipeline] = PipelineFactory()
    mf: BaseFactory[BaseMetric] = MetricFactory()
    sf: BaseFactory[BaseStorage] = StorageFactory()


    @staticmethod
    def bind(manager:Optional[Any] = dict, wrapper:Optional[Any] = dict):
        
        @fundispatch # type: ignore
        def inner(func:Any):
            raise NotImplementedError(f"No decorator registered for {func.__name__}")

        @inner.register(BaseFilter) # type: ignore
        def _(func:Type[BaseFilter]) -> Type[BaseFilter]:
            bind_model = BindGenericModel[BaseFilter](
                manager=manager,
                wrapper=wrapper,
                filter=func
            )
            Container.ff[func.__name__] = bind_model
            return func
        
    
        @inner.register(BasePipeline) # type: ignore
        def _(func:Type[BasePipeline]) -> Type[BasePipeline]:
            bind_model = BindGenericModel[BasePipeline](
                manager=manager,
                wrapper=wrapper,
                filter=func
            )
            Container.pf[func.__name__] = bind_model
            return func
        
        @inner.register(BaseMetric) # type: ignore
        def _(func:Type[BaseMetric]) -> Type[BaseMetric]:
            bind_model = BindGenericModel[BaseMetric](
                manager=manager,
                wrapper=wrapper,
                filter=func
            )
            Container.mf[func.__name__] = bind_model
            return func
        
        @inner.register(BaseStorage) # type: ignore
        def _(func:Type[BaseStorage]) -> Type[BaseStorage]:
            bind_model = BindGenericModel[BaseStorage](
                manager=manager,
                wrapper=wrapper,
                filter=func
            )
            Container.sf[func.__name__] = bind_model
            return func
        
        return inner
    

    