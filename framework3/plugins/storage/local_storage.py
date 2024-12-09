from typing import Any
from framework3.base import BaseStorage
import pickle

# import cloudpickle as pickle
import os
from pathlib import Path

__all__ = ["LocalStorage"]


class LocalStorage(BaseStorage):
    """
    A local file system storage implementation.

    This class provides methods to interact with the local file system for storing and retrieving files.

    Attributes:
        storage_path (str): The base path for storage operations.
        _base_path (str): The full path to the storage directory.

    Example:
        ```python

        >>> storage = LocalStorage(storage_path='my_cache')
        >>> storage.upload_file("Hello, World!", "greeting.txt", "/tmp")
        >>> content = storage.download_file("greeting.txt", "/tmp")
        >>> print(content)
        'Hello, World!'
        ```
    """

    def __init__(self, storage_path: str = "cache"):
        """
        Initialize the LocalStorage.

        Args:
            storage_path (str, optional): The base path for storage. Defaults to 'cache'.

        Example:
            ```python
            >>> storage = LocalStorage(storage_path='my_custom_cache')
            ```
        """
        super().__init__()
        self.storage_path = storage_path
        self._base_path = storage_path

    def get_root_path(self) -> str:
        """
        Get the root path of the storage.

        Returns:
            str: The full path to the storage directory.

        Example:
            ```python

            >>> storage = LocalStorage(storage_path='my_cache')
            >>> print(storage.get_root_path())
            '/home/user/project/my_cache'
            ```
        """
        return self._base_path

    def upload_file(
        self, file, file_name: str, context: str, direct_stream: bool = False
    ):
        """
        Upload a file to the specified context.

        Args:
            file (object): The file content to be uploaded.
            file_name (str): The name of the file.
            context (str): The directory path where the file will be saved.
            direct_stream (bool, optional): Not used in this implementation. Defaults to False.

        Returns:
            (str): The file name if successful, None otherwise.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello, World!", "greeting.txt", "/tmp")
            'greeting.txt'
            ```
        """
        try:
            Path(context).mkdir(parents=True, exist_ok=True)
            print(f"\t * Saving in local path: {context}/{file_name}")
            pickle.dump(file, open(f"{context}/{file_name}", "wb"))
            print("\t * Saved !")
            return file_name
        except Exception as ex:
            print(ex)
        return None

    def list_stored_files(self, context: str):
        """
        List all files in the specified context.

        Args:
            context (str): The directory path to list files from.

        Returns:
            (List[str]): A list of file names in the specified context.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello", "file1.txt", "/tmp")
            >>> storage.upload_file("World", "file2.txt", "/tmp")
            >>> print(storage.list_stored_files("/tmp"))
            ['file1.txt', 'file2.txt']
            ```
        """
        return os.listdir(context)

    def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
        """
        Get a file by its hashcode (filename in this implementation).

        Args:
            hashcode (str): The hashcode (filename) of the file.
            context (str): The directory path where the file is located.

        Returns:
            Any: A file object if found.

        Raises:
            FileNotFoundError: If the file is not found in the specified context.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello", "greeting.txt", "/tmp")
            >>> file = storage.get_file_by_hashcode("greeting.txt", "/tmp")
            >>> print(type(file))
            <class '_io.BufferedReader'>
            ```
        """
        if hashcode in os.listdir(context):
            return open(f"{context}/{hashcode}", "rb")
        else:
            raise FileNotFoundError(f"Couldn't find file {hashcode} in path {context}")

    def check_if_exists(self, hashcode: str, context: str):
        """
        Check if a file exists in the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file.
            context (str): The directory path where to check for the file.

        Returns:
            (bool): True if the file exists, False otherwise.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello", "greeting.txt", "/tmp")
            >>> print(storage.check_if_exists("greeting.txt", "/tmp"))
            True
            >>> print(storage.check_if_exists("nonexistent.txt", "/tmp"))
            False
            ```
        """
        try:
            for file_n in os.listdir(context):
                if file_n == hashcode:
                    return True
            return False
        except FileNotFoundError:
            return False

    def download_file(self, hashcode: str, context: str):
        """
        Download and load a file from the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file to download.
            context (str): The directory path where the file is located.

        Returns:
            (Any): The content of the file, unpickled if it was pickled.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello, World!", "greeting.txt", "/tmp")
            >>> content = storage.download_file("greeting.txt", "/tmp")
            >>> print(content)
            'Hello, World!'
            ```
        """
        stream = self.get_file_by_hashcode(hashcode, context)
        print(f"\t * Downloading: {stream}")
        loaded = pickle.load(stream)
        return pickle.loads(loaded) if isinstance(loaded, bytes) else loaded

    def delete_file(self, hashcode: str, context: str):
        """
        Delete a file from the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file to delete.
            context (str): The directory path where the file is located.

        Raises:
            FileExistsError: If the file does not exist in the specified context.

        Example:
            ```python

            >>> storage = LocalStorage()
            >>> storage.upload_file("Hello", "greeting.txt", "/tmp")
            >>> storage.delete_file("greeting.txt", "/tmp")
            >>> print(storage.check_if_exists("greeting.txt", "/tmp"))
            False
            ```
        """
        if os.path.exists(f"{context}/{hashcode}"):
            os.remove(f"{context}/{hashcode}")
        else:
            raise FileExistsError("No existe en la carpeta")
