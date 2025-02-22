import os
from pydantic_settings import BaseSettings

class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "")

    # aws
    BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "")
    REGION_NAME: str = os.environ.get("REGION_NAME", "")
    ENDPOINT_URL: str = os.environ.get("ENDPOINT_URL", "")
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")


config_settings = Settings()