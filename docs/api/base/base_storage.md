# Base class for storage management

::: framework3.base.base_storage

## Overview

The `BaseStorage` class is an abstract base class that defines the interface for all storage operations in the framework3 project. It inherits from `BasePlugin` for plugin functionality and `BaseSingleton` to ensure only one instance of each storage class is created.

## Key Methods

The `BaseStorage` class defines several abstract methods that must be implemented by any concrete storage class:

- `get_root_path()`: Returns the root path of the storage.
- `upload_file()`: Uploads a file to the storage.
- `download_file()`: Downloads a file from the storage.
- `list_stored_files()`: Lists all files in a specific context.
- `get_file_by_hashcode()`: Retrieves a file using its hashcode.
- `check_if_exists()`: Checks if a file exists in the storage.
- `delete_file()`: Deletes a file from the storage.

## Usage

To create a new storage class, inherit from `BaseStorage` and implement all the abstract methods. For example, see the `LocalStorage` class in `framework3.storage.local_storage` for a concrete implementation.
