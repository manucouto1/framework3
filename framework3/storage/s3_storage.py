import boto3, pickle, io, sys
from typing import Any, List
from botocore.exceptions import ClientError
from framework3.base.base_storage import BaseStorage
from framework3.container.container import Container

@Container.bind()
class S3Storage(BaseStorage):
    def __init__(self, bucket: str, region_name: str, access_key_id: str, access_key: str, endpoint_url: str|None = None):
        super().__init__()
        self._client = boto3.client(
                service_name='s3',
                region_name=region_name,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=access_key,
                endpoint_url=endpoint_url,
                use_ssl=True
        )
        self.bucket = bucket
    
    def get_root_path(self) -> str:
        return self.bucket

    def upload_file(self, file:object, file_name:str, direct_stream:bool=False) -> str:# -> Any | None:
        if type(file) != io.BytesIO:
            binary = pickle.dumps(file)
            stream = io.BytesIO(binary)
        else:
            stream = file
        print("- Binary prepared!")
        
        print("- Stream ready!")
        print(f" \t * Object size {sys.getsizeof(stream) * 1e-9} GBs ")
        self._client.put_object(
            Body=stream,
            Bucket=self.bucket,
            Key=file_name
        )
        print('Upload Complete!')
        return file_name

    def list_stored_files(self) -> List[Any]:# -> Any | None:# -> Any | None:
        return self._client.list_objects_v2(Bucket=self.bucket)['Contents']
   
    def get_file_by_hashcode(self, hashcode:str):
        obj = self._client.get_object(Bucket=self.bucket, Key=hashcode)
        return obj['Body'].read()

    def check_if_exists(self, hashcode:str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=hashcode)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                print(f'An error ocurred > {e}')
                return False
        return True

    def download_file(self, hashcode:str) -> Any:
        obj = self._client.get_object(Bucket=self.bucket, Key=hashcode)
        return pickle.loads(obj['Body'].read())

    def delete_file(self, hashcode:str):
        if self.check_if_exists(hashcode):
            self._client.delete_object(Bucket=self.bucket, Key=hashcode)
            if self.check_if_exists(hashcode):
                raise Exception("Couldn't delete file")
            else:
                print("Deleted!")
        else:
            raise FileExistsError("No existe en el bucket")
        
