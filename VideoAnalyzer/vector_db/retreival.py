from pymilvus import AnnSearchRequest, Collection
from typing import Any, Dict, List, Optional, Union
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever


class CustomMilvusCollectionHybridSearchRetriever(MilvusCollectionHybridSearchRetriever):
    """Custom Hybrid Search Retriever with Partition and Filtering"""

    partition_name: Optional[str] = None  # Added partition support
    filter_expr: Optional[str] = None  # Added filtering support

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Load specific partition if provided
        self.collection.load(
            partition_names=[self.partition_name] if self.partition_name else None
        )

    def _build_ann_search_requests(self, query: str) -> List[AnnSearchRequest]:
        """Override method to include filtering expression"""
        search_requests = []
        for ann_field, embedding, param, limit, expr in zip(
                self.anns_fields,
                self.field_embeddings,
                self.field_search_params,
                self.field_limits,
                self.field_exprs,
        ):
            request = AnnSearchRequest(
                data=[embedding.embed_query(query)],
                anns_field=ann_field,
                param=param,
                limit=limit,
                expr=self.filter_expr if self.filter_expr else expr,  # Apply custom filter if provided
            )
            search_requests.append(request)
        return search_requests