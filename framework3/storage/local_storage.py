from typing import Any
from framework3.base.base_clases import BaseStorage
import pickle
import os


class LocalStorage(BaseStorage):
    def __init__(self, storage_path:str='data'):
        super().__init__()
        self.storage_path=storage_path
        
    def upload_file(self, file, file_name, direct_stream=False):
        try:
            print(f"* Saving in local path: {self.storage_path}/{file_name}")
            pickle.dump(file, open(f'{self.storage_path}/{file_name}', 'wb'))
            print("* Saved !")
            return file_name
        except Exception as ex:
            print(ex)
        return None
    
    def list_stored_files(self):
        return os.listdir(self.storage_path)
    
    def get_file_by_hashcode(self, hashcode:str) -> Any:
        print(hashcode)
        print(os.listdir(self.storage_path))
        if hashcode in os.listdir(self.storage_path):
            return open(f'{self.storage_path}/{hashcode}', 'rb')
        else:
            raise FileNotFoundError(f"Couldn't find file {hashcode} in path {self.storage_path}")
                
    def check_if_exists(self, hashcode:str):
        for file_n in os.listdir(self.storage_path):
            if file_n == hashcode:
                return True
        return False
            
    def download_file(self, hashcode:str):
        stream = self.get_file_by_hashcode(hashcode)
        print(f"*************************************** ------------------- STREMA > {stream}")
        return pickle.load(stream)
    
    def delete_file(self, hashcode:str):
        if os.path.exists(f'{self.storage_path}/{hashcode}'):
            os.remove(f'{self.storage_path}/{hashcode}')
        else:
            raise FileExistsError("No existe en la carpeta")