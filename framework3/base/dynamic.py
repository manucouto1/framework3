from typing import Callable, Any, Type

class DynamicTypedClass(type):
    def __new__(cls, name: str, bases: tuple, dct: dict):
        # En este punto podemos modificar la clase, añadir campos, métodos, etc.
        return super().__new__(cls, name, bases, dct)

    @classmethod
    def create(cls, base_cls: Type[Any], class_name: str) -> Type[Any]:
        # Crear una clase dinámica a partir de un base_cls
        new_class = type(class_name, (base_cls,), {})

        # Aquí podemos añadir comportamiento adicional como métodos dinámicos o atributos
        return new_class