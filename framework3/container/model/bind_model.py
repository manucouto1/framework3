from pydantic import BaseModel, ConfigDict, Field
from typing import Callable, Generic, Type, Any, TypeVar, Union
from framework3.base.base_types import TypePlugable

class BindGenericModel(BaseModel, Generic[TypePlugable]):
    manager: Any | None = None
    wrapper: Any | None = None
    filter: Type[TypePlugable] = Field(..., exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed = True,
        populate_by_name = True
    )