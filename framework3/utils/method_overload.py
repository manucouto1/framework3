from functools import singledispatch, update_wrapper
from typing import Any, Callable, Protocol, TypeVar, cast
from functools import wraps


T = TypeVar("T")
R = TypeVar("R")  # Tipo de retorno de las funciones registradas

class DispatchableMethod(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...
    def register(self, cls: type[T], func: Callable[..., R]) -> Callable[..., R]: ...

class SingleDispatch(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...
    def register(self, cls: type, func: Callable[..., R]) -> Callable[..., R]: ...
    def dispatch(self, cls: type) -> Callable[..., R]: ...


def methdispatch(func: Callable[..., R]) -> DispatchableMethod[R]:
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        # Despacho basado en el tipo del segundo argumento (args[1])
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    
    wrapper = cast(DispatchableMethod[R], wrapper)
    wrapper.register = dispatcher.register  # Agregar el método register
    update_wrapper(wrapper, func)           # Preservar metadatos
    return wrapper


# def fundispatch(func: Callable[..., R]) -> SingleDispatch[R]:
#     dispatcher = singledispatch(func)
    
#     def wrapper(*args, **kw):
#         # Despacho basado en el tipo del segundo argumento (args[0])
#         # Obtener el tipo del segundo argumento
#         arg_type = args[0] if isinstance(args[0], type) else type(args[0])
#         # Despachar basado en el tipo
#         return dispatcher.dispatch(arg_type)(*args, **kw)

#     wrapper = cast(SingleDispatch[R], wrapper)
#     wrapper.register = dispatcher.register
#     wrapper.dispatch = dispatcher.dispatch
#     update_wrapper(wrapper, func)           # Preservar metadatos
#     return wrapper

def fundispatch(func: SingleDispatch[R]) -> SingleDispatch[R]:
    dispatcher = singledispatch(func)
    
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Determinar el tipo del primer argumento (args[0])
        arg_type = args[0] if isinstance(args[0], type) else type(args[0])
        # Usar el despachador para llamar a la función registrada
        return dispatcher.dispatch(arg_type)(*args, **kwargs)

    # Asociar las funciones de registro y despacho del dispatcher original
    wrapper = cast(SingleDispatch[R], wrapper)
    wrapper.register = dispatcher.register
    wrapper.dispatch = dispatcher.dispatch
    # Preservar metadatos y anotaciones de tipo de la función original
    update_wrapper(wrapper, func)
    return wrapper