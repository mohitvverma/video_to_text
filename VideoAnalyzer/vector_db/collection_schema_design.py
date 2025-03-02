from langchain_ollama import OllamaEmbeddings
from VideoAnalyzer.settings import config_settings
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from chat_with_summarization.vector_db.extract_dense_emb_dimension_size import dense_embedding_length

schema = None


def get_collection_schema():
    try:
        fields =[
            FieldSchema(
                name=settings.PRIMARY_KEY_FIELD_SCHEMA_NAME,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=settings.COLLECTION_SCHEMA_AUTO_ID_STATUS,
                max_length=settings.COLLECTION_SCHEMA_MAX_LENGTH,
            ),
            FieldSchema(name=settings.DENSE_FIELD_SCHEMA_NAME, dtype=DataType.FLOAT_VECTOR, dim=dense_embedding_length),
            FieldSchema(name=settings.SPARSE_FIELD_SCHEMA_NAME, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=settings.TEXT_FIELD_SCHEMA_NAME, dtype=DataType.VARCHAR, max_length=settings.SCHEMA_MAX_LENGTH),
            FieldSchema(name=settings.PARTITION_FIELD_SCHEMA_NAME, dtype=DataType.JSON, max_length=settings.SCHEMA_MAX_LENGTH),
            FieldSchema(name=settings.TIMESTAMP_FIELD_SCHEMA_NAME, dtype=DataType.JSON, max_length=settings.SCHEMA_MAX_LENGTH),
            FieldSchema(name=settings.COLLECTION_FIELD_SCHEMA_NAME, dtype=DataType.JSON, max_length=settings.SCHEMA_MAX_LENGTH)
        ]

        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        return schema

    except Exception as e:
        logger.error(f"Error {e}")
        return e

try:
    schema_design = get_collection_schema()
    logger.debug(f"Schema Created : \n{schema_design}")
except Exception as e:
    logger.error(f"Error while creating schema {e}")