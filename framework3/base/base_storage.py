from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Any, Type

from framework3.base import BasePlugin

__all__ = ["BaseStorage", "BaseSingleton"]


class BaseSingleton:
    """
    A base class for implementing the Singleton pattern.

    This class ensures that only one instance of each derived class is created.

    Key Features:
        - Implements the Singleton design pattern
        - Allows derived classes to have only one instance

    Usage:
        To create a Singleton class, inherit from BaseSingleton:

        ```python
        class MySingleton(BaseSingleton):
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1

        # Usage
        instance1 = MySingleton()
        instance2 = MySingleton()
        assert instance1 is instance2  # True
        ```

    Attributes:
        _instances (Dict[Type[BaseSingleton], Any]): A class-level dictionary to store instances.

    Methods:
        __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
            Creates a new instance or returns the existing one.

    Note:
        This class should be used as a base class for any class that needs to implement
        the Singleton pattern.
    """

    _instances: Dict[Type[BaseSingleton], Any] = {}
    _verbose: bool = True

    def __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
        """
        Create a new instance of the class if it doesn't exist, otherwise return the existing instance.

        This method implements the core logic of the Singleton pattern.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            BaseStorage: The single instance of the class.

        Note:
            This method is called before __init__ when creating a new instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)  # type: ignore
        return cls._instances[cls]


class BaseStorage(BasePlugin, BaseSingleton):
    """
    An abstract base class for storage operations.

    This class defines the interface for storage-related operations and inherits
    from BasePlugin for plugin functionality and BaseSingleton for single instance behavior.

    Key Features:
        - Abstract methods for common storage operations
        - Singleton behavior ensures only one instance per storage type
        - Inherits plugin functionality from BasePlugin

    Usage:
        To create a new storage type, inherit from BaseStorage and implement all abstract methods:

        ```python
        class MyCustomStorage(BaseStorage):
            def __init__(self, root_path: str):
                self.root_path = root_path

            def get_root_path(self) -> str:
                return self.root_path

            def upload_file(self, file, file_name: str, context: str, direct_stream: bool = False) -> str | None:
                # Implement file upload logic
                ...

            # Implement other abstract methods
            ...

        # Usage
        storage = MyCustomStorage("/path/to/storage")
        storage.upload_file(file_object, "example.txt", "documents")
        ```

    Methods:
        get_root_path() -> str:
            Abstract method to get the root path of the storage.
        upload_file(file: object, file_name: str, context: str, direct_stream: bool = False) -> str | None:
            Abstract method to upload a file to the storage.
        download_file(hashcode: str, context: str) -> Any:
            Abstract method to download a file from the storage.
        list_stored_files(context: str) -> List[Any]:
            Abstract method to list all files stored in a specific context.
        get_file_by_hashcode(hashcode: str, context: str) -> Any:
            Abstract method to retrieve a file by its hashcode.
        check_if_exists(hashcode: str, context: str) -> bool:
            Abstract method to check if a file exists in the storage.
        delete_file(hashcode: str, context: str):
            Abstract method to delete a file from the storage.

    Note:
        This is an abstract base class. Concrete implementations should override
        all abstract methods to provide specific storage functionality.
    """

    @abstractmethod
    def get_root_path(self) -> str:
        """
        Get the root path of the storage.

        This method should be implemented to return the base directory or path
        where the storage system keeps its files.

        Returns:
            str: The root path of the storage.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def get_root_path(self) -> str:
                return "/var/data/storage"
            ```
        """
        ...

    @abstractmethod
    def upload_file(
        self, file: object, file_name: str, context: str, direct_stream: bool = False
    ) -> str | None:
        """
        Upload a file to the storage.

        This method should be implemented to handle file uploads to the storage system.

        Args:
            file (object): The file object to upload.
            file_name (str): The name of the file.
            context (str): The context or directory for the file.
            direct_stream (bool, optional): Whether to use direct streaming. Defaults to False.

        Returns:
            str | None: The identifier of the uploaded file, or None if upload failed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def upload_file(self, file: object, file_name: str, context: str, direct_stream: bool = False) -> str | None:
                path = os.path.join(self.get_root_path(), context, file_name)
                with open(path, 'wb') as f:
                    f.write(file.read())
                return file_name
            ```
        """
        ...

    @abstractmethod
    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download a file from the storage.

        This method should be implemented to retrieve files from the storage system.

        Args:
            hashcode (str): The identifier of the file to download.
            context (str): The context or directory of the file.

        Returns:
            Any: The downloaded file object.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def download_file(self, hashcode: str, context: str) -> Any:
                path = os.path.join(self.get_root_path(), context, hashcode)
                with open(path, 'rb') as f:
                    return f.read()
            ```
        """
        ...

    @abstractmethod
    def list_stored_files(self, context: str) -> List[Any]:
        """
        List all files stored in a specific context.

        This method should be implemented to return a list of files in a given context.

        Args:
            context (str): The context or directory to list files from.

        Returns:
            List[Any]: A list of file objects or file information.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def list_stored_files(self, context: str) -> List[Any]:
                path = os.path.join(self.get_root_path(), context)
                return os.listdir(path)
            ```
        """
        ...

    @abstractmethod
    def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
        """
        Retrieve a file by its hashcode.

        This method should be implemented to fetch a specific file using its identifier.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            Any: The file object or file information.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
                return self.download_file(hashcode, context)
            ```
        """
        ...

    @abstractmethod
    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in the storage.

        This method should be implemented to verify the existence of a file.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            bool: True if the file exists, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def check_if_exists(self, hashcode: str, context: str) -> bool:
                path = os.path.join(self.get_root_path(), context, hashcode)
                return os.path.exists(path)
            ```
        """
        ...

    @abstractmethod
    def delete_file(self, hashcode: str, context: str):
        """
        Delete a file from the storage.

        This method should be implemented to remove a file from the storage system.

        Args:
            hashcode (str): The identifier of the file to delete.
            context (str): The context or directory of the file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def delete_file(self, hashcode: str, context: str):
                path = os.path.join(self.get_root_path(), context, hashcode)
                if os.path.exists(path):
                    os.remove(path)
                else:
                    raise FileNotFoundError(f"File {hashcode} not found in {context}")
            ```
        """
        ...
