import boto3
from pathlib import Path


def upload_file_to_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client("s3")
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"{local_path} not found.")
    s3.upload_file(str(path), bucket_name, s3_key)
    print(f"Uploaded {path} to s3://{bucket_name}/{s3_key}")
