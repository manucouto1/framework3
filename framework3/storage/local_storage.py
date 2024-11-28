from typing import Any
from framework3.base.base_storage import BaseStorage
import pickle
import os
from pathlib import Path


class LocalStorage(BaseStorage):
    def __init__(self, storage_path:str='cache'):
        super().__init__()
        self.storage_path=storage_path
        self._base_path = f'{os.getcwd()}/{self.storage_path}'

    def get_root_path(self) -> str:
        return self._base_path
        
    def upload_file(self, file, file_name, context, direct_stream=False):
        try:
            Path(context).mkdir(parents=True, exist_ok=True)
            print(f"\t * Saving in local path: {context}/{file_name}")
            pickle.dump(file, open(f'{context}/{file_name}', 'wb'))
            print("\t * Saved !")
            return file_name
        except Exception as ex:
            print(ex)
        return None
    
    def list_stored_files(self, context:str):
        return os.listdir(context)
    
    def get_file_by_hashcode(self, hashcode:str, context:str) -> Any:
        if hashcode in os.listdir(context):
            return open(f'{context}/{hashcode}', 'rb')
        else:
            raise FileNotFoundError(f"Couldn't find file {hashcode} in path {context}")
                
    def check_if_exists(self, hashcode:str, context:str):
        try:
            for file_n in os.listdir(context):
                if file_n == hashcode:
                    return True
            return False
        except FileNotFoundError:
            return False
            
    def download_file(self, hashcode:str, context:str):
        stream = self.get_file_by_hashcode(hashcode, context)
        print(f"\t * Downloading: {stream}")
        loaded = pickle.load(stream)
        return pickle.loads(loaded) if isinstance(loaded, bytes) else loaded 
    
    def delete_file(self, hashcode:str, context):
        if os.path.exists(f'{context}/{hashcode}'):
            os.remove(f'{context}/{hashcode}')
        else:
            raise FileExistsError("No existe en la carpeta")