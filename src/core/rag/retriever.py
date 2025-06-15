from typing import List, Dict, Any, Optional
import concurrent.futures
from core.config import settings
from qdrant_client import models
from sentence_transformers.SentenceTransformer import SentenceTransformer
import gc
import psutil
import os

import core.logger_utils as logger_utils
from core import lib
from core.db.qdrant import QdrantDatabaseConnector
from core.rag.query_expanison import QueryExpansion
from core.rag.reranking import Reranker
from core.rag.self_query import SelfQuery

logger = logger_utils.get_logger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Current memory usage: {memory_usage_mb:.2f} MB")

class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and collection-based search.
    """

    COLLECTION_TYPES = [
        "edgar_fillings",
        "earnings_call_transcripts",
        "financial_news",
        "investopedia",
        "macro_economic_reports"
    ]

    def __init__(self, query: str) -> None:
        self._client = None  # Lazy loading
        self.query = query
        self._embedder = None  # Lazy loading
        self._query_expander = None  # Lazy loading
        self._metadata_extractor = None  # Lazy loading
        self._reranker = None  # Lazy loading
        self._max_workers = int(os.getenv("MAX_WORKERS", 2))
        self._batch_size = int(os.getenv("BATCH_SIZE", 32))

    def _get_client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            self._client = QdrantDatabaseConnector()
        return self._client

    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        return self._embedder

    def _get_query_expander(self):
        """Lazy load query expander."""
        if self._query_expander is None:
            self._query_expander = QueryExpansion()
        return self._query_expander

    def _get_metadata_extractor(self):
        """Lazy load metadata extractor."""
        if self._metadata_extractor is None:
            self._metadata_extractor = SelfQuery()
        return self._metadata_extractor

    def _get_reranker(self):
        """Lazy load reranker."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _get_vector_collection_name(self, data_type: str) -> str:
        """Get the vector collection name for a given data type."""
        return f"vector_{data_type}"

    def _search_single_query(
        self,
        generated_query: str,
        collection_type: Optional[str] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Any]:
        """Search a single query across specified collections with filters."""
        try:
            # Get embedding in smaller batches
            query_vector = self._get_embedder().encode(
                generated_query,
                batch_size=self._batch_size,
                show_progress_bar=False
            ).tolist()
            
            search_filter = self._construct_search_query(
                collection_type=collection_type,
                additional_filters=additional_filters
            )
            
            # Search across all collections
            all_results = []
            for data_type in self.COLLECTION_TYPES:
                collection_name = self._get_vector_collection_name(data_type)
                try:
                    logger.info(
                        "Searching in collection",
                        collection_name=collection_name,
                        query=generated_query
                    )
                    results = self._get_client().search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        query_filter=search_filter,
                        limit=k // len(self.COLLECTION_TYPES)
                    )
                    logger.info(
                        "Search results from collection",
                        collection_name=collection_name,
                        num_results=len(results)
                    )
                    all_results.extend(results)
                    
                    # Clean up after each collection search
                    gc.collect()
                    log_memory_usage()
                    
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
        to_expand_to_n_queries: int = 3,
        collection_type: str | None = None,
        additional_filters: dict | None = None,
    ) -> list:
        """Retrieve top k documents using query expansion."""
        try:
            # Log initial memory usage
            log_memory_usage()
            
            # Extract date range and modify query if needed
            metadata = self._get_metadata_extractor().extract_metadata(self.query)
            self.query = metadata["modified_query"]
            
            logger.info(
                "Starting query expansion",
                query=self.query
            )
            
            # Generate expanded queries
            generated_queries = self._get_query_expander().generate_response(
                self.query,
                to_expand_to_n=to_expand_to_n_queries
            )
            logger.info(
                "Generated queries for search",
                num_queries=len(generated_queries),
                queries=generated_queries
            )
            
            # Search using each generated query with limited workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                search_tasks = [
                    executor.submit(
                        self._search_single_query,
                        query,
                        collection_type,
                        additional_filters,
                        k
                    )
                    for query in generated_queries
                ]

                hits = []
                for future in concurrent.futures.as_completed(search_tasks):
                    result = future.result()
                    hits.extend(result)
                    # Clean up after each query
                    gc.collect()
                    log_memory_usage()

            # Sort all hits by score and take top k
            hits.sort(key=lambda x: x.score, reverse=True)
            hits = hits[:k]

            logger.info(
                "Retrieved documents",
                num_documents=len(hits)
            )

            return hits
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
        keep_top_k: int = 5
    ) -> List[str]:
        """Rerank the retrieved documents."""
        try:
            # Process content in batches
            content_list = [hit.payload["content"] for hit in hits]
            rerank_hits = self._get_reranker().generate_response(
                query=self.query,
                passages=content_list,
                keep_top_k=keep_top_k
            )
            
            # Clean up after reranking
            gc.collect()
            log_memory_usage()
            
            return rerank_hits
        except Exception as e:
            logger.error(
                "Error in rerank",
                error=str(e)
            )
            return []

    def set_query(self, query: str):
        """Set a new query."""
        self.query = query
        # Clean up resources when setting new query
        self._embedder = None
        gc.collect()
        log_memory_usage()

