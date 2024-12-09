from typing import List
from framework3.base import BaseDatasetManager, VData, XYData
from framework3.container.container import Container


@Container.bind()
class DatasetManager(BaseDatasetManager):
    """
    Manages datasets for training and evaluating models.
    """

    def list(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List[str]: A list of dataset names.
        """
        return Container.storage.list_stored_files(
            f"{Container.storage.get_root_path()}/datasets"
        )

    def save(self, name: str, data: VData) -> None:
        """
        Save a dataset.

        Args:
            name (str): Unique name for the dataset.
            data (XYData): The data to be saved.

        Raises:
            ValueError: If a dataset with the given name already exists.
        """
        if Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):  # type: ignore
            raise ValueError(f"Dataset '{name}' already exists.")
        Container.storage.upload_file(
            data, name, f"{Container.storage.get_root_path()}/datasets"
        )

    def update(self, name: str, data: VData) -> None:
        """
        Update a dataset.

        Args:
            name (str): Unique name for the dataset.
            data (XYData): The data to be saved.

        Raises:
            ValueError: If a dataset with the given name doesn't exists.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):  # type: ignore
            raise ValueError(f"Dataset '{name}' does not exist.")
        Container.storage.upload_file(
            data, name, f"{Container.storage.get_root_path()}/datasets"
        )

    def load(self, name: str) -> XYData:
        """
        Load a dataset.

        Args:
            name (str): Name of the dataset to load.

        Returns:
            XYData: The loaded dataset.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):
            raise ValueError(f"Dataset '{name}' does not exist.")
        return XYData(
            _hash=name,
            _path="datasets",
            _value=lambda: Container.storage.download_file(
                name, f"{Container.storage.get_root_path()}/datasets"
            ),
        )

    def delete(self, name: str) -> None:
        """
        Delete a dataset.

        Args:
            name (str): Name of the dataset to delete.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):
            raise ValueError(f"Dataset '{name}' does not exist.")
        Container.storage.delete_file(
            name, f"{Container.storage.get_root_path()}/datasets"
        )
