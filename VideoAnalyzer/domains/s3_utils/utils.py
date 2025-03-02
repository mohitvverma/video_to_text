import boto3
from langchain_core.documents import Document
import botocore.exceptions
from typing import Any, BinaryIO
from loguru import logger


def get_s3_client(
        region_name: str,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
) -> Any:
    return boto3.session.Session().client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def upload_to_spaces(
        client: Any, file: BinaryIO, bucket_name: str, file_name: str, content_type: str
) -> None:
    try:
        client.upload_fileobj(
            file,
            bucket_name,
            file_name,
            ExtraArgs={"ContentType": content_type},
        )
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        logger.exception("Failed to upload to spaces")
        raise Exception(f"Failed to upload to spaces: {e}") from e

    logger.info(f"File uploaded to spaces: {file_name}")
