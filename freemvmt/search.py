"""
Redis Vector Search implementation for Two-Towers document retrieval model.

This module provides functionality to:
1. Build a Redis vector search index with document embeddings
2. Search for nearest neighbors given a query
3. Handle document ingestion and query processing
"""

import argparse
import os
from typing import Any, Optional

import torch
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from model import TwoTowersModel
from training import MSMarcoDataset


def create_index_schema(projection_dim: int = 128) -> IndexSchema:
    """Create Redis vector search index schema."""
    schema_dict = {
        "index": {
            "name": "document_index",
            "prefix": "doc:",
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "id",
                "type": "tag",
            },
            {
                "name": "content",
                "type": "text",
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": projection_dim,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
    return IndexSchema.from_dict(schema_dict)


class DocumentSearchEngine:
    """Redis-based document search engine using vector similarity."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        projection_dim: int = 128,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "document_index",
    ):
        """
        Initialize the search engine.

        Args:
            model_path: Path to saved model weights (optional)
            projection_dim: Dimension of the model's projection layer
            redis_url: Redis connection URL
            index_name: Name of the Redis search index
        """
        self.projection_dim = projection_dim
        self.redis_url = redis_url
        self.index_name = index_name

        # Initialize Redis client
        self.redis_client = Redis.from_url(redis_url)

        # Initialize model
        self.model = TwoTowersModel(projection_dim=projection_dim)
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            print("Using untrained model (random weights)")

        self.model.eval()

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize search index
        self.search_index: Optional[SearchIndex] = None
        self._setup_search_index()

    def _setup_search_index(self):
        """Set up Redis search index with the appropriate schema."""
        schema = create_index_schema(self.projection_dim)

        try:
            # Try to create the index
            self.search_index = SearchIndex(schema=schema)
            self.search_index.connect(redis_url=self.redis_url)

            # Check if index already exists, if not create it
            if not self.search_index.exists():
                print(f"Creating new search index: {self.index_name}")
                self.search_index.create(overwrite=False)
            else:
                print(f"Using existing search index: {self.index_name}")

        except Exception as e:
            print(f"Error setting up search index: {e}")
            raise

    def ingest_documents(
        self,
        documents: list[str],
        batch_size: int = 1024,
        clear_existing: bool = False,
    ):
        """
        Ingest documents into the Redis vector store.

        Args:
            documents: list of document texts to ingest
            batch_size: Batch size for processing documents
            clear_existing: Whether to clear existing documents before ingesting
        """
        if clear_existing:
            print("Clearing existing documents from index...")
            if self.search_index:
                self.search_index.clear()

        print(f"Ingesting {len(documents)} documents...")

        # Encode all documents in batches
        with torch.no_grad():
            all_embeddings = self.model.encode_documents_batched(
                documents=documents,
                batch_size=batch_size,
            )

        # Convert embeddings to numpy for Redis storage
        embeddings_np = all_embeddings.cpu().numpy()

        # Prepare documents for bulk insertion
        data = []
        for i, (doc_text, embedding) in enumerate(zip(documents, embeddings_np)):
            data.append(
                {
                    "id": str(i),
                    "content": doc_text,
                    "embedding": embedding.tobytes(),  # Convert to bytes for Redis storage
                }
            )

        # Bulk insert documents
        print("Inserting documents into Redis...")
        if self.search_index:
            self.search_index.load(data, id_field="id")
            print(f"✅ Successfully ingested {len(documents)} documents")
        else:
            raise RuntimeError("Search index not initialized")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Search for similar documents given a query.

        Args:
            query: Query text to search for
            top_k: Number of top results to return

        Returns:
            list of dictionaries containing document info and similarity scores
        """
        # Encode the query
        with torch.no_grad():
            query_embedding = self.model.encode_queries([query])
            query_vector = query_embedding.cpu().numpy()[0]  # Get first (and only) result

        # Create vector query
        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=["id", "content"],
            num_results=top_k,
        )

        # Execute search
        if not self.search_index:
            raise RuntimeError("Search index not initialized")

        results = self.search_index.query(vector_query)

        # Format results
        formatted = []
        for result in results:
            formatted.append(
                {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "score": float(result.get("vector_score", 0.0)),
                    "distance": 1.0 - float(result.get("vector_score", 0.0)),  # convert similarity -> distance
                }
            )
        return formatted

    def get_index_info(self) -> dict[str, Any]:
        """Get information about the current index."""
        if not self.search_index:
            return {"error": "Search index not initialized"}

        try:
            info = self.search_index.info()
            return {
                "index_name": info.get("index_name"),
                "num_docs": info.get("num_docs"),
                "indexing_failures": info.get("indexing_failures"),
                "vector_index_sz": info.get("vector_index_sz"),
            }
        except Exception as e:
            return {"error": str(e)}


def build_document_index(
    max_docs: int = 10000,
    batch_size: int = 1024,
    model_path: Optional[str] = None,
    projection_dim: int = 128,
):
    """Build the document index from MS Marco dataset."""
    print("Building document index from MS Marco dataset...")

    # Initialize search engine
    engine = DocumentSearchEngine(
        model_path=model_path,
        projection_dim=projection_dim,
    )

    # Load dataset and get unique documents
    print(f"Loading MS Marco dataset (max {max_docs} documents)...")
    dataset = MSMarcoDataset("train", max_samples=max_docs)
    unique_docs = dataset.get_unique_passages()

    # Limit to max_docs if specified
    if max_docs > 0 and len(unique_docs) > max_docs:
        unique_docs = unique_docs[:max_docs]

    print(f"Found {len(unique_docs)} unique documents")

    # Ingest documents
    engine.ingest_documents(unique_docs, batch_size=batch_size, clear_existing=True)

    # Print index info
    info = engine.get_index_info()
    print(f"Index info: {info}")

    return engine


def search_documents(query: str, top_k: int = 10, model_path: Optional[str] = None):
    """Search for documents similar to the given query."""
    print(f"Searching for: '{query}'")
    print("-" * 50)

    # Initialize search engine
    engine = DocumentSearchEngine(model_path=model_path)

    # Check if index exists and has documents
    info = engine.get_index_info()
    if info.get("num_docs", 0) == 0:
        print("❌ No documents found in index. Please run with --build-index first.")
        return

    print(f"Searching index with {info.get('num_docs', 0)} documents...")

    # Perform search
    results = engine.search(query, top_k=top_k)

    # Display results
    print(f"\n🔍 Top {len(results)} results:")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
        print(f"{i}. [Score: {result['score']:.4f}] [Distance: {result['distance']:.4f}]")
        print(f"   {content_preview}")
        if i < len(results):
            print()


def main():
    """Main CLI interface for the search system."""
    parser = argparse.ArgumentParser(description="Redis Vector Search for Two-Towers Model")

    # Main actions
    parser.add_argument("query", nargs="?", help="Query to search for")
    parser.add_argument("--build-index", action="store_true", help="Build the document index")
    parser.add_argument("--index-info", action="store_true", help="Show index information")

    # Configuration
    parser.add_argument("--model-path", type=str, help="Path to saved model weights")
    parser.add_argument("--projection-dim", type=int, default=128, help="Model projection dimension")
    parser.add_argument("--max-docs", type=int, default=10000, help="Maximum documents to index")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL")

    args = parser.parse_args()

    # Handle different actions
    if args.build_index:
        build_document_index(
            max_docs=args.max_docs,
            batch_size=args.batch_size,
            model_path=args.model_path,
            projection_dim=args.projection_dim,
        )
    elif args.index_info:
        engine = DocumentSearchEngine(model_path=args.model_path, projection_dim=args.projection_dim)
        info = engine.get_index_info()
        print(f"Index Information: {info}")
    elif args.query:
        search_documents(
            query=args.query,
            top_k=args.top_k,
            model_path=args.model_path,
        )
    else:
        # Default demo query if no arguments provided
        demo_query = "machine learning algorithms for text classification"
        print("No query provided. Running demo search...")
        search_documents(query=demo_query, top_k=args.top_k, model_path=args.model_path)


if __name__ == "__main__":
    main()
