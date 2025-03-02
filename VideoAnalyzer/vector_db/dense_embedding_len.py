import asyncio
from VideoAnalyzer.vector_db.utils import get_embedding_model
from VideoAnalyzer.settings import config_settings
from loguru import logger

get_embedding_dimension = None


def get_embedding_dimension_singleton():
    """
    A function-based Singleton to get and cache the embedding dimension.
    No recalculation is performed after the first successful computation.
    """
    _cached_dimension = None  # Closure variable to store the dimension

    async def get_dimension():
        nonlocal _cached_dimension  # Access the cached variable
        if _cached_dimension is not None:
            logger.info("Using cached embedding dimension.")
            return _cached_dimension

        try:
            logger.info("Calculating embedding dimension...")
            embed_model = await get_embedding_model()
            embedding = embed_model.embed_query("Test Embedding Dimension")

            _cached_dimension = len(embedding)  # Cache the result
            logger.info(f"Embedded dimension computed: {_cached_dimension}")
            return _cached_dimension

        except Exception as e:
            logger.error(f"Error while calculating embedding dimension: {e}")
            raise

    return get_dimension


# Create a Singleton function for embedding dimension
get_embedding_dimension = get_embedding_dimension_singleton()


async def main():
    try:
        dense_embedding_length = await get_embedding_dimension()
        print(f"Dense Embedding Length: {dense_embedding_length}")
        return dense_embedding_length
    except Exception as e:
        logger.error(f"Error: Unable to fetch embedding dimension. {e}")

# Example usage
try:
    dense_embedding_length = asyncio.run(main())
except Exception as e:
    logger.error(f"Error: Unable to fetch embedding dimension. {e}")
