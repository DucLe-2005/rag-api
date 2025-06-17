from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import datetime

import core.logger_utils as logger_utils

from core.rag.retriever import VectorRetriever
from core.rag.self_query import SelfQuery

import uvicorn

logger = logger_utils.get_logger(__name__)

app = FastAPI(
    title="Financial RAG Pipeline API",
    description="API for accessing the financial RAG pipeline functionality",
    version="1.0.0"
)

# Configure CORS with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8888", "https://finbud.pro/", "https://finbud-ai.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

class QueryRequest(BaseModel):
    query: str
    collection_type: Optional[str] = None
    additional_filters: Optional[Dict[str, Any]] = None
    top_k: int = 3
    expand_n_query: int = 2
    keep_top_k: int = 3

@app.post("/api/query")
async def process_query(request: QueryRequest) -> Dict[str, List[str]]:
    try:
        # Initialize retriever with query
        try:
            retriever = VectorRetriever(query=request.query)
        except Exception as e:
            logger.error(f"Error initializing retriever: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error connecting to vector database. Please check your Qdrant connection settings."
            )
        
        # Retrieve relevant documents
        try:
            hits = retriever.retrieve_top_k(
                k=request.top_k,
                to_expand_to_n_queries=request.expand_n_query,
                collection_type=None,  # None means search all collections
                additional_filters=request.additional_filters
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error retrieving documents from vector database."
            )
        
        # Rerank the hits
        try:
            context = retriever.rerank(hits=hits, keep_top_k=request.keep_top_k)
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            context = []
        
        return {"context": context}
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing query: {str(e)}"
        )

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    try:
        # Basic health check without Qdrant connection
        return {
            "status": "healthy",
            "api": "ok",
            "timestamp": str(datetime.datetime.now())
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "api": "ok",
            "error": str(e)
        }

@app.get("/api/collection-types")
async def get_collection_types() -> Dict[str, List[str]]:
    """Get available collection types for filtering."""
    return {"collection_types": VectorRetriever.COLLECTION_TYPES}

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    # Use 0.0.0.0 to bind to all available network interfaces
    uvicorn.run(app, host="0.0.0.0", port=port) 