from typing import Dict, Iterator, Tuple, Type, Generic
from framework3.base.base_types import TypePlugable
from rich import print as rprint

__all__ = ["BaseFactory"]


class BaseFactory(Generic[TypePlugable]):
    """
    A generic factory class for managing and creating pluggable components.

    This class provides a flexible way to register, retrieve, and manage
    different types of components (plugins) in the framework.

    Example:
    ```python

    from framework3.base.base_factory import BaseFactory
    from framework3.base.base_plugin import BasePlugin

    class MyComponentFactory(BaseFactory[BasePlugin]):
        pass

    factory = MyComponentFactory()
    factory['ComponentA'] = ComponentA
    factory['ComponentB'] = ComponentB

    component_a = factory['ComponentA']()
    component_b = factory['ComponentB']()
    ```
    """

    def __init__(self):
        """
        Initialize the BaseFactory with an empty dictionary to store components.
        """
        self._foundry: Dict[str, Type[TypePlugable]] = {}

    def __getattr__(self, name: str) -> Type[TypePlugable]:
        """
        Retrieve a component by attribute access.

        Args:
            name (str): The name of the component to retrieve.

        Returns:
            Type[TypePlugable]: The requested component class.

        Raises:
            AttributeError: If the component is not found.
        """
        if name in self._foundry:
            return self._foundry[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Type[TypePlugable]) -> None:
        """
        Set a component by attribute assignment.

        Args:
            name (str): The name to assign to the component.
            value (Type[TypePlugable]): The component class to register.
        """
        if name == "_foundry":
            super().__setattr__(name, value)
        else:
            self._foundry[name] = value

    def __setitem__(self, name: str, value: Type[TypePlugable]) -> None:
        """
        Set a component using dictionary-like syntax.

        Args:
            name (str): The name to assign to the component.
            value (Type[TypePlugable]): The component class to register.
        """
        if name == "_foundry":
            super().__setattr__(name, value)
        else:
            self._foundry[name] = value

    def __getitem__(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable]:
        """
        Retrieve a component using dictionary-like syntax.

        Args:
            name (str): The name of the component to retrieve.
            default (Type[TypePlugable] | None, optional): Default value if component is not found.

        Returns:
            Type[TypePlugable]: The requested component class or the default value.

        Raises:
            AttributeError: If the component is not found and no default is provided.
        """
        if name in self._foundry:
            return self._foundry[name]
        else:
            if default is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
            return default

    def __iter__(self) -> Iterator[Tuple[str, Type[TypePlugable]]]:
        """
        Provide an iterator over the registered components.

        Returns:
            Iterator[Tuple[str, Type[TypePlugable]]]: An iterator of (name, component) pairs.
        """
        return iter(self._foundry.items())

    def __contains__(self, item: str) -> bool:
        """
        Check if a component is registered in the factory.

        Args:
            item (str): The name of the component to check.

        Returns:
            bool: True if the component is registered, False otherwise.
        """
        return item in self._foundry

    def get(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable]:
        """
        Retrieve a component by name.

        Args:
            name (str): The name of the component to retrieve.
            default (Type[TypePlugable] | None, optional): Default value if component is not found.

        Returns:
            Type[TypePlugable]: The requested component class or the default value.

        Raises:
            AttributeError: If the component is not found and no default is provided.
        """
        if name in self._foundry:
            return self._foundry[name]
        else:
            if default is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
            return default

    def print_available_components(self):
        """
        Print a list of all available components in the factory.

        This method uses rich formatting to display the components in a visually appealing way.
        """
        rprint(f"[bold]Available {self.__class__.__name__[:-7]}s:[/bold]")
        for name, binding in self._foundry.items():
            rprint(f"  - [green]{name}[/green]: {binding}")
