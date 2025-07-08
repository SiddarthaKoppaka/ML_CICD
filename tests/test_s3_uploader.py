# tests/test_s3_uploader.py
from unittest.mock import patch
from training_pipeline.s3_uploader import upload_file_to_s3


@patch("training_pipeline.s3_uploader.boto3.client")
def test_upload_file_to_s3(mock_boto):
    mock_client = mock_boto.return_value
    mock_client.upload_file.return_value = None  # simulate success
    upload_file_to_s3("models/model_v1.pkl", "dummy-bucket", "models/model_v1.pkl")
    mock_client.upload_file.assert_called_once()
