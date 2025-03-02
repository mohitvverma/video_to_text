import os
from pydantic_settings import BaseSettings
from typing import ClassVar


class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "")
    MILVUS_COLLECTION_NAME_DEV: str = os.environ.get("MILVUS_COLLECTION_NAME_DEV", "")
    MILVUS_DROP_COLLECTION_STATUS: bool = os.environ.get("MILVUS_DROP_COLLECTION_STATUS", "")

    # Modular LLM Names Settings
    LLM_SERVICE: str = os.environ.get("LLM_SERVICE", "openai")
    OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "")
    MILVUS_COLLECTION_NAME_PROD: str = os.environ.get("MILVUS_COLLECTION_NAME_PROD", "")
    MILVUS_COLLECTION_NAME_TEST: str = os.environ.get("MILVUS_COLLECTION_NAME_TEST", "")
    MILVUS_COLLECTION_NAME_STAGING: str = os.environ.get("MILVUS_COLLECTION_NAME_STAGING", "")

    # chunk settings
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", 100))
    INITIAL_NUMBER_OF_PAGES_TO_RETRIEVE_FOR_SUMMARIZATION: int = int(
        os.environ.get("INITIAL_NUMBER_OF_PAGES_TO_RETRIEVE_FOR_SUMMARIZATION", 5)
    )

    # aws
    BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "")
    REGION_NAME: str = os.environ.get("REGION_NAME", "")
    ENDPOINT_URL: str = os.environ.get("ENDPOINT_URL", "")
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    SUMMARIZE_LLM_MODEL: str = os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o")

    # Modular LLM Names
    LLMS: ClassVar[dict] = {
        "RAG_LLM_MODEL": os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o"),
        "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "AUDIO_LLM_MODEL": os.environ.get("AUDIO_LLM_MODEL", "whisper-1"),
    }

    AZURE_OPENAI_SETTINGS: ClassVar[dict] = {
        "LLM_MODEL_NAME": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_LLM_MODEL_NAME", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_LLM_MODEL_NAME", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_LLM_MODEL_NAME", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_LLM_MODEL_NAME", ""),
        },

        "OPENAI_AUDIO_TRANSCRIPTION_MODEL": {
            "ENDPOINT": os.environ.get(
                "AZURE_ENDPOINT_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "API_KEY": os.environ.get(
                "AZURE_API_KEY_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
        },
    }


    OLLAMA_LLM_SETTING: ClassVar[dict] = {
        "RAG_LLM_MODEL": os.environ.get("RAG_LLM_MODEL", "llama3.1:latest"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "llama3.1:latest"),
        "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "AUDIO_LLM_MODEL": os.environ.get("AUDIO_LLM_MODEL", "whisper-1"),
    }

config_settings = Settings()
