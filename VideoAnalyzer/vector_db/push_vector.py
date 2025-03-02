

def push_to_database(texts, index_name, namespace):
    # collect metadata to track source information
    meta_datas = [text['metadata'] for text in texts]

    Pinecone.from_texts(
        [t['page_content'] for t in texts],
        get_embeddings(model_key="EMBEDDING_MODEL"),
        meta_datas,
        index_name=index_name,
        namespace=namespace,
    )

    # logger.info("Vectors have been pushed to database")