# Local Storage Management

The Local Storage module in Framework3 provides a simple and efficient way to store and retrieve data on the local file system. This module is particularly useful for caching, persisting models, and managing intermediate results in data processing pipelines.

## Overview

The `LocalStorage` class implements the `BaseStorage` interface, offering a set of methods for file operations on the local file system. It provides functionality for uploading, downloading, listing, and deleting files, as well as checking for file existence.

## Key Features

- File upload and download
- Directory creation and management
- File existence checking
- File listing
- File deletion
- Support for direct file streaming

## Usage

### Basic Usage

To use the Local Storage, first import and instantiate the `LocalStorage` class:

```python
from framework3.plugins.storage.local_storage import LocalStorage

# Initialize LocalStorage with a root path
storage = LocalStorage(storage_path='/path/to/storage/root')
```

### Uploading Files

```python
content = "Hello, World!"
file_name = "greeting.txt"
context = "messages"

# Upload a file
storage.upload_file(file=content, file_name=file_name, context=context)
```

### Downloading Files

```python
# Download a file
downloaded_content = storage.download_file(file_name, context)
print(downloaded_content)  # Outputs: Hello, World!
```

### Checking File Existence

```python
# Check if a file exists
exists = storage.check_if_exists(file_name, context)
print(exists)  # Outputs: True
```

### Listing Files

```python
# List files in a context
files = storage.list_stored_files(context)
print(files)  # Outputs: ['greeting.txt']
```

### Deleting Files

```python
# Delete a file
storage.delete_file(file_name, context)
```

## Advanced Usage

### Direct File Streaming

For large files or when memory efficiency is crucial, you can use direct file streaming:

```python
with open('large_file.dat', 'rb') as file:
    storage.upload_file(file=file, file_name='large_file.dat', context='data', direct_stream=True)
```

### Custom File Naming

You can implement custom file naming strategies by subclassing `LocalStorage` and overriding the `_get_file_name` method:

```python
class CustomLocalStorage(LocalStorage):
    def _get_file_name(self, file_name: str) -> str:
        # Custom file naming logic
        return f"custom_prefix_{file_name}"
```

## Best Practices

1. **Path Management**: Always use the `get_root_path()` method to ensure correct path handling across different operating systems.
2. **Error Handling**: Implement proper error handling to manage file operation failures gracefully.
3. **Context Usage**: Organize your files using contexts (subdirectories) for better file management.
4. **Regular Cleanup**: Implement a strategy to clean up old or unused files to manage storage space effectively.
5. **Security Considerations**: Be cautious about the permissions set on the storage directory and the files within it.

## API Reference

For a detailed look at the `LocalStorage` class and its methods, refer to the auto-generated documentation below:

::: framework3.plugins.storage.local_storage.LocalStorage

## Conclusion

The Local Storage module in Framework3 provides a robust and flexible solution for managing file operations on the local file system. By leveraging this module, you can easily integrate local file storage capabilities into your data processing pipelines, caching mechanisms, and model persistence strategies.
