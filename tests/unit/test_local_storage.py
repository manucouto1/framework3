import pytest
import os
import pickle
from framework3.storage.local_storage import LocalStorage
from framework3.base.base_clases import BaseStorage

@pytest.fixture
def local_storage():
    storage_path = 'test_data'
    os.makedirs(storage_path, exist_ok=True)
    storage = LocalStorage(storage_path=storage_path)
    yield storage
    for file in os.listdir(storage_path):
        os.remove(os.path.join(storage_path, file))
    os.rmdir(storage_path)


def test_upload_file(local_storage):
    test_data = {'key': 'value'}
    file_name = 'test_file.pkl'
    
    result = local_storage.upload_file(test_data, file_name)
    
    assert result == file_name
    assert os.path.exists(os.path.join(local_storage.storage_path, file_name))
    
    with open(os.path.join(local_storage.storage_path, file_name), 'rb') as f:
        loaded_data = pickle.load(f)
    
    assert loaded_data == test_data

def test_upload_file_exception(local_storage, mocker):
    test_data = {'key': 'value'}
    file_name = 'test_file.pkl'
    
    # Mock the pickle.dump function to raise an exception
    mocker.patch('pickle.dump', side_effect=Exception("Mocked error"))
    
    result = local_storage.upload_file(test_data, file_name)
    
    assert result is None
    assert os.path.exists(os.path.join(local_storage.storage_path, file_name))

def test_list_stored_files(local_storage):
    # Create some test files
    test_files = ['file1.txt', 'file2.pkl', 'file3.bin']
    for file_name in test_files:
        with open(os.path.join(local_storage.storage_path, file_name), 'w') as f:
            f.write('test content')
    
    # Call the method to list stored files
    stored_files = local_storage.list_stored_files()
    
    # Check if the returned list contains all the test files
    assert set(stored_files) == set(test_files)
    
    # Check if the number of files matches
    assert len(stored_files) == len(test_files)
    
    # Clean up the test files
    for file_name in test_files:
        os.remove(os.path.join(local_storage.storage_path, file_name))

def test_get_file_by_hashcode(local_storage):
    # Create a test file
    test_file_name = 'test_file.txt'
    test_content = b'Test content'
    with open(os.path.join(local_storage.storage_path, test_file_name), 'wb') as f:
        f.write(test_content)

    # Retrieve the file using get_file_by_hashcode
    retrieved_file = local_storage.get_file_by_hashcode(test_file_name)

    # Assert that the retrieved file is a file object
    assert hasattr(retrieved_file, 'read')

    # Read the content of the retrieved file
    retrieved_content = retrieved_file.read()

    # Assert that the content matches the original
    assert retrieved_content == test_content

    # Close the file object
    retrieved_file.close()

    # Clean up the test file
    os.remove(os.path.join(local_storage.storage_path, test_file_name))

def test_get_non_existent_file(local_storage):
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileNotFoundError) as excinfo:
        local_storage.get_file_by_hashcode(non_existent_file)
    assert str(excinfo.value) == f"Couldn't find file {non_existent_file} in path {local_storage.storage_path}"

def test_check_if_exists_true(local_storage):
    # Create a test file
    test_file = 'test_file.txt'
    test_content = 'Test content'
    with open(os.path.join(local_storage.storage_path, test_file), 'w') as f:
        f.write(test_content)

    # Check if the file exists using its name as hashcode
    result = local_storage.check_if_exists(test_file)

    # Assert that the file exists
    assert result is True

    # Clean up the test file
    os.remove(os.path.join(local_storage.storage_path, test_file))


def test_check_if_exists_false(local_storage):
    non_existent_file = 'non_existent_file.txt'
    result = local_storage.check_if_exists(non_existent_file)
    assert result is False

def test_download_file(local_storage):
    # Create a test file
    test_data = {'key': 'value'}
    file_name = 'test_file.pkl'
    
    # Upload the test file
    local_storage.upload_file(test_data, file_name)
    
    # Download and deserialize the file
    downloaded_data = local_storage.download_file(file_name)
    
    # Assert that the downloaded data matches the original data
    assert downloaded_data == test_data
    
    # Clean up the test file
    os.remove(os.path.join(local_storage.storage_path, file_name))

def test_delete_existing_file(local_storage):
    # Create a test file
    test_file = 'test_file.txt'
    test_content = 'Test content'
    with open(os.path.join(local_storage.storage_path, test_file), 'w') as f:
        f.write(test_content)

    # Delete the file
    local_storage.delete_file(test_file)

    # Assert that the file no longer exists
    assert not os.path.exists(os.path.join(local_storage.storage_path, test_file))

    # Verify that check_if_exists returns False
    assert local_storage.check_if_exists(test_file) is False

def test_delete_non_existent_file(local_storage):
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileExistsError) as excinfo:
        local_storage.delete_file(non_existent_file)
    assert str(excinfo.value) == "No existe en la carpeta"