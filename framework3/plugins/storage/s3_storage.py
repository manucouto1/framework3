import boto3
import pickle
import io
import sys
from typing import Any, List
from botocore.exceptions import ClientError
from framework3.base import BaseStorage
from framework3.container import Container

__all__ = ["S3Storage"]


@Container.bind()
class S3Storage(BaseStorage):
    """
    A storage implementation for Amazon S3.

    This class provides methods to interact with Amazon S3 for storing and retrieving files.

    Attributes:
        _client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.

    Example:
    ```python
        >>> storage = S3Storage(bucket='my-bucket', region_name='us-west-2',
        ...                     access_key_id='YOUR_ACCESS_KEY', access_key='YOUR_SECRET_KEY')
        >>> storage.upload_file("Hello, World!", "greeting.txt", "my-folder")
        >>> content = storage.download_file("greeting.txt", "my-folder")
        >>> print(content)
        'Hello, World!'
    ```
    """

    def __init__(
        self,
        bucket: str,
        region_name: str,
        access_key_id: str,
        access_key: str,
        endpoint_url: str | None = None,
    ):
        """
        Initialize the S3Storage.

        Args:
            bucket (str): The name of the S3 bucket.
            region_name (str): The AWS region name.
            access_key_id (str): The AWS access key ID.
            access_key (str): The AWS secret access key.
            endpoint_url (str|None, optional): The endpoint URL for the S3 service. Defaults to None.

        Example:
            >>> storage = S3Storage(bucket='my-bucket', region_name='us-west-2',
            ...                     access_key_id='YOUR_ACCESS_KEY', access_key='YOUR_SECRET_KEY')
        """
        super().__init__()
        self._client = boto3.client(
            service_name="s3",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key,
            endpoint_url=endpoint_url,
            use_ssl=True,
        )
        self.bucket = bucket

    def get_root_path(self) -> str:
        """
        Get the root path (bucket name) of the storage.

        Returns:
            str: The name of the S3 bucket.

        Example:
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> print(storage.get_root_path())
            'my-bucket'
        """
        return self.bucket

    def upload_file(
        self, file: object, file_name: str, context: str, direct_stream: bool = False
    ) -> str:
        """
        Upload a file to the specified context in S3.

        Args:
            file (object): The file content to be uploaded.
            file_name (str): The name of the file.
            context (str): The directory path where the file will be saved.
            direct_stream (bool, optional): If True, assumes file is already a BytesIO object. Defaults to False.

        Returns:
            str: The file name if successful.

        Example:
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> storage.upload_file("Hello, World!", "greeting.txt", "my-folder")
            'greeting.txt'
        """
        if type(file) is not io.BytesIO:
            binary = pickle.dumps(file)
            stream = io.BytesIO(binary)
        else:
            stream = file
        print("- Binary prepared!")

        print("- Stream ready!")
        print(f" \t * Object size {sys.getsizeof(stream) * 1e-9} GBs ")
        self._client.put_object(
            Body=stream, Bucket=self.bucket, Key=f"{context}/{file_name}"
        )
        print("Upload Complete!")
        return file_name

    def list_stored_files(self, context) -> List[Any]:
        """
        List all files in the specified context.

        Args:
            context (str): Not used in this implementation.

        Returns:
            (List[Any]): A list of dictionaries containing information about the objects in the bucket.

        Example:
        ```python
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> files = storage.list_stored_files("")
            >>> for file in files:
            ...     print(file['Key'])
        ```
        """
        return list(
            map(
                lambda x: x["Key"],
                self._client.list_objects_v2(Bucket=self.bucket)["Contents"],
            )
        )

    def get_file_by_hashcode(self, hashcode: str, context: str):
        """
        Get a file by its hashcode (key in S3).

        Args:
            hashcode (str): The hashcode (key) of the file.
            context (str): Not used in this implementation.

        Returns:
            (bytes): The content of the file.

        Example:
        ```python
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> content = storage.get_file_by_hashcode("my-folder/greeting.txt", "")
            >>> print(content.decode())
            'Hello, World!'
        ```
        """
        obj = self._client.get_object(Bucket=self.bucket, Key=hashcode)
        return obj["Body"].read()

    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Returns:
            bool: True if the file exists, False otherwise.

        Example:
        ```python
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> exists = storage.check_if_exists("greeting.txt", "my-folder")
            >>> print(exists)
            True
        ```
        """
        try:
            self._client.head_object(Bucket=self.bucket, Key=f"{context}/{hashcode}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                print(f"An error ocurred > {e}")
                return False
        return True

    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download a file from S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Returns:
            Any: The deserialized content of the file.

        Example:
        ```python
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> content = storage.download_file("greeting.txt", "my-folder")
            >>> print(content)
            'Hello, World!'
        ```
        """
        obj = self._client.get_object(Bucket=self.bucket, Key=f"{context}/{hashcode}")
        return pickle.loads(obj["Body"].read())

    def delete_file(self, hashcode: str, context: str) -> None:
        """
        Delete a file from S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Raises:
            Exception: If the file couldn't be deleted.
            FileExistsError: If the file doesn't exist in the bucket.

        Example:
        ```python
            >>> storage = S3Storage(bucket='my-bucket', ...)
            >>> storage.delete_file("greeting.txt", "my-folder")
            Deleted!
        ```
        """
        if self.check_if_exists(hashcode, context):
            self._client.delete_object(Bucket=self.bucket, Key=f"{context}/{hashcode}")
            if self.check_if_exists(hashcode, context):
                raise Exception("Couldn't delete file")
            else:
                print("Deleted!")
        else:
            raise FileExistsError("No existe en el bucket")
