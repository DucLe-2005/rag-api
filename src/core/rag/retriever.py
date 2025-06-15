from typing import List, Dict, Any, Optional
import concurrent.futures
from core.config import settings
from qdrant_client import models
from sentence_transformers.SentenceTransformer import SentenceTransformer
import gc

import core.logger_utils as logger_utils
from core import lib
from core.db.qdrant import QdrantDatabaseConnector
from core.rag.query_expanison import QueryExpansion
from core.rag.reranking import Reranker
from core.rag.self_query import SelfQuery

logger = logger_utils.get_logger(__name__)

class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and collection-based search.
    Memory-optimized version for deployment with limited RAM.
    """

    COLLECTION_TYPES = [
        "edgar_fillings",
        "earnings_call_transcripts",
        "financial_news",
        "investopedia",
        "macro_economic_reports"
    ]

    def __init__(self, query: str) -> None:
        self._client = QdrantDatabaseConnector()
        self.query = query
        self._embedder = None  # Lazy loading
        self._query_expander = QueryExpansion()
        self._metadata_extractor = SelfQuery()
        self._reranker = Reranker()

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        self._embedder = None
        gc.collect()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        return self._embedder

    def _get_vector_collection_name(self, data_type: str) -> str:
        """Get the vector collection name for a given data type."""
        return f"vector_{data_type}"

    def _search_single_query(
        self,
        generated_query: str,
        collection_type: Optional[str] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
        k: int = 3
    ) -> List[Any]:
        """Search a single query across specified collections with filters."""
        try:
            query_vector = self._get_embedder().encode(generated_query).tolist()
            search_filter = self._construct_search_query(
                collection_type=collection_type,
                additional_filters=additional_filters
            )
            
            # Search across all collections sequentially to reduce memory usage
            all_results = []
            for data_type in self.COLLECTION_TYPES:
                collection_name = self._get_vector_collection_name(data_type)
                try:
                    results = self._client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        query_filter=search_filter,
                        limit=k // len(self.COLLECTION_TYPES)
                    )
                    all_results.extend(results)
                    # Clear results after extending to free memory
                    results = None
                except Exception as e:
                    logger.error(
                        "Error searching collection",
                        collection_name=collection_name,
                        error=str(e)
                    )
                    continue
            return all_results
        except Exception as e:
            logger.error(
                "Error in _search_single_query",
                error=str(e),
                query=generated_query
            )
            return []

    def _construct_search_query(
        self,
        collection_type: Optional[str] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> models.Filter:
        """Create a filter for Qdrant search based on collection type and additional filters."""
        must_conditions = []
        
        if collection_type:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=collection_type)
                )
            )
        
        if additional_filters:
            for key, value in additional_filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=must_conditions) if must_conditions else None

    def retrieve_top_k(
        self,
        k: int = 3,
        to_expand_to_n_queries: int = 2,
        collection_type: str | None = None,
        additional_filters: dict | None = None,
    ) -> list:
        """Retrieve top k documents using query expansion."""
        try:
            # Extract date range and modify query if needed
            metadata = self._metadata_extractor.extract_metadata(self.query)
            self.query = metadata["modified_query"]
            
            # Generate expanded queries
            generated_queries = self._query_expander.generate_response(
                self.query,
                to_expand_to_n=to_expand_to_n_queries
            )
            
            # Process queries sequentially instead of concurrently
            all_hits = []
            for query in generated_queries:
                hits = self._search_single_query(
                    query,
                    collection_type,
                    additional_filters,
                    k
                )
                all_hits.extend(hits)
                # Clear hits after extending to free memory
                hits = None

            # Sort all hits by score and take top k
            all_hits.sort(key=lambda x: x.score, reverse=True)
            result = all_hits[:k]
            
            # Clear all_hits to free memory
            all_hits = None
            gc.collect()
            
            return result
        except Exception as e:
            logger.error(
                "Error in retrieve_top_k",
                error=str(e),
                query=self.query
            )
            return []

    def rerank(
        self,
        hits: List[Dict[str, Any]],
        keep_top_k: int = 3
    ) -> List[str]:
        """Rerank the retrieved documents."""
        try:
            content_list = [hit.payload["content"] for hit in hits]
            result = self._reranker.generate_response(
                query=self.query,
                passages=content_list,
                keep_top_k=keep_top_k
            )
            
            # Clear content_list to free memory
            content_list = None
            gc.collect()
            
            return result
        except Exception as e:
            logger.error(
                "Error in rerank",
                error=str(e)
            )
            return []

    def set_query(self, query: str):
        self.query = query

