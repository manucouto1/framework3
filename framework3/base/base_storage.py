from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Any, Type

from framework3.base import BasePlugin

__all__ = ["BaseStorage", "BaseSingleton"]


class BaseSingleton:
    """
    A base class for implementing the Singleton pattern.

    This class ensures that only one instance of each derived class is created.
    """

    _instances: Dict[Type[BaseSingleton], Any] = {}

    def __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
        """
        Create a new instance of the class if it doesn't exist, otherwise return the existing instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            BaseStorage: The single instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)  # type: ignore
        return cls._instances[cls]


class BaseStorage(BasePlugin, BaseSingleton):
    """
    An abstract base class for storage operations.

    This class defines the interface for storage-related operations and inherits
    from BasePlugin for plugin functionality and BaseSingleton for single instance behavior.

    Example:
    ```python

    from framework3.base import BaseStorage
    import os

    class SimpleFileStorage(BaseStorage):
        def __init__(self, root_path):
            self.root_path = root_path

        def get_root_path(self) -> str:
            return self.root_path

        def upload_file(self, file, file_name: str, context: str, direct_stream: bool = False) -> str | None:
            full_path = os.path.join(self.root_path, context, file_name)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(file)
            return file_name

        def download_file(self, hashcode: str, context: str):
            full_path = os.path.join(self.root_path, context, hashcode)
            with open(full_path, 'rb') as f:
                return f.read()

        def list_stored_files(self, context: str):
            full_path = os.path.join(self.root_path, context)
            return os.listdir(full_path)

        def get_file_by_hashcode(self, hashcode: str, context: str):
            return self.download_file(hashcode, context)

        def check_if_exists(self, hashcode: str, context: str) -> bool:
            full_path = os.path.join(self.root_path, context, hashcode)
            return os.path.exists(full_path)

        def delete_file(self, hashcode: str, context: str):
            full_path = os.path.join(self.root_path, context, hashcode)
            if os.path.exists(full_path):
                os.remove(full_path)
            else:
                raise FileNotFoundError(f"File {hashcode} not found in {context}")

    # Usage
    storage = SimpleFileStorage("/tmp/my_storage")
    storage.upload_file(b"Hello, World!", "greeting.txt", "messages")
    content = storage.download_file("greeting.txt", "messages")
    print(content)  # b'Hello, World!'


    ```
    """

    @abstractmethod
    def get_root_path(self) -> str:
        """
        Get the root path of the storage.

        Returns:
            str: The root path of the storage.
        """
        ...

    @abstractmethod
    def upload_file(
        self, file: object, file_name: str, context: str, direct_stream: bool = False
    ) -> str | None:
        """
        Upload a file to the storage.

        Args:
            file (object): The file object to upload.
            file_name (str): The name of the file.
            context (str): The context or directory for the file.
            direct_stream (bool, optional): Whether to use direct streaming. Defaults to False.

        Returns:
            str | None: The identifier of the uploaded file, or None if upload failed.
        """
        ...

    @abstractmethod
    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download a file from the storage.

        Args:
            hashcode (str): The identifier of the file to download.
            context (str): The context or directory of the file.

        Returns:
            Any: The downloaded file object.
        """
        ...

    @abstractmethod
    def list_stored_files(self, context: str) -> List[Any]:
        """
        List all files stored in a specific context.

        Args:
            context (str): The context or directory to list files from.

        Returns:
            List[Any]: A list of file objects or file information.
        """
        ...

    @abstractmethod
    def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
        """
        Retrieve a file by its hashcode.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            Any: The file object or file information.
        """
        ...

    @abstractmethod
    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in the storage.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        ...

    @abstractmethod
    def delete_file(self, hashcode: str, context: str):
        """
        Delete a file from the storage.

        Args:
            hashcode (str): The identifier of the file to delete.
            context (str): The context or directory of the file.
        """
        ...
