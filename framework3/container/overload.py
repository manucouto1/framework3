from functools import singledispatch, update_wrapper
from typing import Any, Callable, Protocol, TypeVar, cast
from functools import wraps

T = TypeVar("T")
R = TypeVar("R")  # Return type of registered functions

class DispatchableMethod(Protocol[R]):
    """
    Protocol for a dispatchable method.

    This protocol defines the interface for a method that can be dispatched
    based on the type of its arguments and can register new implementations.
    """
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """Call the method with the given arguments."""
        ...

    def register(self, cls: type[T], func: Callable[..., R]) -> Callable[..., R]:
        """Register a new implementation for the given class."""
        ...

class SingleDispatch(Protocol[R]):
    """
    Protocol for a single dispatch function.

    This protocol defines the interface for a function that can be dispatched
    based on the type of its first argument and can register new implementations.
    """
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """Call the function with the given arguments."""
        ...

    def register(self, cls: type, func: Callable[..., R]) -> Callable[..., R]:
        """Register a new implementation for the given class."""
        ...

    def dispatch(self, cls: type) -> Callable[..., R]:
        """Return the implementation for the given class."""
        ...

def methdispatch(func: Callable[..., R]) -> DispatchableMethod[R]:
    """
    Decorator for creating a method dispatch.

    This decorator creates a wrapper around the given function that dispatches
    based on the type of the second argument (typically 'self' in method calls).

    Args:
        func (Callable[..., R]): The function to be wrapped.

    Returns:
        DispatchableMethod[R]: A wrapper function with dispatch capabilities.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        # Dispatch based on the type of the second argument (args[1])
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    
    wrapper = cast(DispatchableMethod[R], wrapper)
    wrapper.register = dispatcher.register  # Add the register method
    update_wrapper(wrapper, func)           # Preserve metadata
    return wrapper

def fundispatch(func: SingleDispatch[R]) -> SingleDispatch[R]:
    """
    Decorator for creating a function dispatch.

    This decorator creates a wrapper around the given function that dispatches
    based on the type of the first argument.

    Args:
        func (SingleDispatch[R]): The function to be wrapped.

    Returns:
        SingleDispatch[R]: A wrapper function with dispatch capabilities.

    Example:
    ```python

    @fundispatch # type: ignore
        def inner(func:Any):
            raise NotImplementedError(f"No decorator registered for {func.__name__}")

        @inner.register(BaseFilter) # type: ignore
        def _(func:Type[BaseFilter]) -> Type[BaseFilter]:
            Container.ff[func.__name__] = func
            Container.pif[func.__name__] = func
            return func
    
        @inner.register(BasePipeline) # type: ignore
        def _(func:Type[BasePipeline]) -> Type[BasePipeline]:
            Container.pf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func
        
        @inner.register(BaseMetric) # type: ignore
        def _(func:Type[BaseMetric]) -> Type[BaseMetric]:
            Container.mf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func
        
        @inner.register(BaseStorage) # type: ignore
        def _(func:Type[BaseStorage]) -> Type[BaseStorage]:
            Container.sf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func
        
        return inner
    ```
    """
    dispatcher = singledispatch(func)
    
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Determine the type of the first argument (args[0])
        arg_type = args[0] if isinstance(args[0], type) else type(args[0])
        # Use the dispatcher to call the registered function
        return dispatcher.dispatch(arg_type)(*args, **kwargs)

    # Associate the register and dispatch functions from the original dispatcher
    wrapper = cast(SingleDispatch[R], wrapper)
    wrapper.register = dispatcher.register
    wrapper.dispatch = dispatcher.dispatch
    # Preserve metadata and type annotations from the original function
    update_wrapper(wrapper, func)
    return wrapper