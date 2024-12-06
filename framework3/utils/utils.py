import inspect


def method_is_overridden(cls, method_name):
    method = getattr(cls, method_name)
    for base in inspect.getmro(cls)[1:]:  # Skip the class itself
        if hasattr(base, method_name):
            return method is not getattr(base, method_name)
    return False


# def method_is_overridden(cls, method_name):
#     return getattr(cls, method_name) != getattr(cls.__base__, method_name, None)


# def method_is_overridden(cls, method_name):
#     return method_name in cls.__dict__


# def method_is_overridden(obj, method_name):
#     method = getattr(obj, method_name)
#     return isinstance(method, types.MethodType) and method.__self__.__class__ is type(obj)
