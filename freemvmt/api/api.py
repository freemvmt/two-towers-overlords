"""
FastAPI backend for two-towers document retrieval system.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Determine if running in Docker
if os.path.exists("/.dockerenv"):
    # if so, import modules from attached volume (rather than local filesystem)
    print("Running in Docker environment, importing modules from /app volume")
    import sys

    sys.path.append("/app")
    from search import DocumentSearchEngine
else:
    from ..search import DocumentSearchEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and search engine
search_engine: Optional[DocumentSearchEngine] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    id: str
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    index_info: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the search engine on startup."""
    global search_engine

    try:
        print("Initializing document search engine...")

        # Get configuration from environment
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        print(f"Redis URL: {redis_url}")

        # Initialize search engine
        search_engine = DocumentSearchEngine(
            redis_url=redis_url,
        )

        # Check if index exists
        index_info = search_engine.get_index_info()
        print(f"Redis index info: {index_info}")

        if index_info.get("num_docs", 0) == 0:
            print("Redis index appears to be empty. Make sure to build the index first!")
        else:
            print(f"Successfully connected to Redis with {index_info.get('num_docs', 0)} documents")

    except Exception as e:
        print(f"Failed to initialize search engine: {e}")
        raise

    yield

    # Cleanup
    print("Shutting down search engine...")


# Create FastAPI app with lifespan events
api = FastAPI(
    title="Two Towers Document Retrieval API",
    description="API for semantic document search using two-towers architecture",
    version="1.0.0",
    lifespan=lifespan,
)


@api.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with Redis connectivity status."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        index_info = search_engine.get_index_info()
        redis_connected = True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        index_info = {"error": str(e)}
        redis_connected = False

    return HealthResponse(
        status="healthy" if redis_connected else "unhealthy", redis_connected=redis_connected, index_info=index_info
    )


@api.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents similar to the query."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        import time

        start_time = time.time()

        # Perform search
        raw_results = search_engine.search(request.query, top_k=request.top_k)

        # Convert to response format
        results = [
            SearchResult(id=result["id"], content=result["content"], score=float(result["score"]))
            for result in raw_results
        ]

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return SearchResponse(
            query=request.query, results=results, total_results=len(results), processing_time_ms=processing_time
        )

    except Exception as e:
        print(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@api.get("/index-info")
async def get_index_info():
    """Get information about the current search index."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        return search_engine.get_index_info()
    except Exception as e:
        print(f"Failed to get index info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index info: {str(e)}")


# Root endpoint
@api.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Two-Towers Document Retrieval API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "search": "/search", "index_info": "/index-info"},
    }
