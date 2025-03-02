from VideoAnalyzer.vector_db.models import MilvusConnectionRequest
from langchain_openai import OpenAIEmbeddings
from VideoAnalyzer.settings import config_settings
from pymilvus import utility, DataType
from loguru import logger
import asyncio
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: list[Document], CHUNK_SIZE: int = 500, CHUNK_OVERLAP: int=100) -> list[str]:
    """
    Splits a list of documents into smaller chunks using a recursive character text splitter.
    """
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts=text_splitter.split_documents(text)
    logger.info(f"Split text into {len(texts)} chunks")
    return texts


def validate_sparse_embedding(sparse_embedding: dict[int, float]) -> bool:
    """
    Custom validation function to check if a sparse embedding is valid.
    Returns True if the sparse embedding is non-empty; False otherwise.
    """
    logger.debug("Validating Sparse Embedding")
    return len(sparse_embedding) > 0


def create_index_with_type(collection=None):
    """
    Creates an index in a collection with the given type of field.

    The method takes a collection object and creates an index on the given field.
    The type of index created depends on the type of field. For example, for a
    dense field, a dense index is created and for a sparse field, a sparse index
    is created.

    If an index already exists for a given field, the method will skip its creation.

    The method returns a boolean indicating whether the index creation was successful.

    Parameters:
    collection (Collection): The collection object in which the index is to be created.

    Returns:
    bool: True if the index creation was successful; False otherwise.
    """
    status = False
    try:
        # Iterate over the collection schema fields
        for field in collection.schema.fields:
            field_name = field.name
            field_type = field.dtype

            # Check and create Dense index
            if field_type == DataType.FLOAT_VECTOR:
                if any(index.field_name == field_name for index in collection.indexes):
                    logger.warning(f"Index already exists for Dense field '{field_name}'. Skipping its creation.")

                else:
                    logger.debug(f"Creating Dense index in the collection with Index Type {config_settings.DENSE_INDEX_TYPE} "
                                 f"& Metric Type {config_settings.DENSE_METRIC_TYPE}")
                    dense_index = {"index_type": config_settings.DENSE_INDEX_TYPE, "metric_type": config_settings.DENSE_METRIC_TYPE}
                    collection.create_index(field_name, dense_index)

            # Check and create Sparse index
            elif field_type == DataType.SPARSE_FLOAT_VECTOR:
                if any(index.field_name == field_name for index in collection.indexes):
                    logger.warning(f"Index already exists for Sparse field '{field_name}'. Skipping its creation.")
                else:
                    logger.debug(
                        f"Creating Sparse index in the collection with Index Type {config_settings.SPARSE_INDEX_TYPE} "
                        f"& Metric Type {config_settings.SPARSE_METRIC_TYPE}")
                    sparse_index = {"index_type": config_settings.SPARSE_INDEX_TYPE,
                                    "metric_type": config_settings.SPARSE_METRIC_TYPE}
                    collection.create_index(field_name, sparse_index)
    except Exception as e:
        logger.error(f"Error while creating index: {e}")
        raise e

    finally:
        collection.flush()
        status = True
        return status


def validate_collection_name(collection_name: str):
    try:
        if utility.has_collection(collection_name):
            return True
        else:
            return False
    except Exception as e:
        logger.error("Collection Name issue")
        raise e


async def get_embedding_model():
    try:
        embed_model=OpenAIEmbeddings(
            model=config_settings.OPENAI_EMBEDDING_MODEL,
            api_key=config_settings.OPENAI_API_KEY
        )
        return embed_model

    except Exception as e:
        logger.error(f"Error {e}")


try:
    dense_embed_func=asyncio.run(get_embedding_model())
    logger.info("Dense Embedding Model Loaded Successfully")

except Exception as e:
    logger.error(f"Error {e}")
    raise e
