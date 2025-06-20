"""
Redis Vector Search implementation for Two-Towers document retrieval model.

This module provides functionality to:
1. Build a Redis vector search index with document embeddings
2. Search for nearest neighbors given a query
3. Handle document ingestion and query processing

Interface Design:
- Two distinct modes: INDEX BUILDING and SEARCH
- INDEX BUILDING: python search.py --build-index [--model-path MODEL]
- SEARCH: python search.py "query text"
- Model consistency enforced: --model-path only allowed with --build-index
- Auto-model selection: uses best trained model by default
"""

import argparse
import os
import random
import re
from typing import Any, Optional

import torch
from tqdm import tqdm
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema


from model import TwoTowersModel
from data import MSMarcoDataset


MODELS_DIR = "models"
WEIGHTS_OVERRIDE = "weights.pt"
DEFAULT_INDEX_NAME = "default_index"
DEFAULT_PROJ_DIM = 128


def find_best_model(models_dir: str = MODELS_DIR) -> Optional[tuple[str, str]]:
    """
    Find the model file with the highest number of epochs, unless there is a file
    with the name given by WEIGHTS_OVERRIDE present, in which case that file is returned directly.

    Args:
        models_dir: Directory containing model files (relative to freemvmt/)

    Returns:
        Path to the best model file, or None if no models found
    """
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        print(f"❌ Models directory '{models_dir}' does not exist or is not a directory")
        return None

    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith(".pt"):
            # Extract epoch from filename like "e9.lr3.d512.m3.pt" (see MODEL_FILENAME_TEMPLATE in main.py)
            match = re.match(r"^e(\d+)", filename)
            if match:
                epochs = int(match.group(1))
                path = os.path.join(models_dir, filename)
                model_files.append((epochs, path, filename))
        elif filename in (WEIGHTS_OVERRIDE):
            print(f"🔍 Best model selection overridden by presence of {WEIGHTS_OVERRIDE} file in {models_dir}")
            return os.path.join(models_dir, filename), filename

    if not model_files:
        print(f"ℹ️ No trained models found in '{models_dir}'")
        return None

    # Sort by epochs in descending order and return the best model
    model_files.sort(key=lambda x: x[0], reverse=True)
    best_epochs, best_path, best_filename = model_files[0]

    print(f"🎯 Auto-selected best model: {best_filename} (trained for {best_epochs} epochs)")
    return best_path, best_filename


def get_projection_dim_from_model(model_path: str) -> int:
    """
    Extract projection dimension from saved model state dict
    (rather than require it be passed w/ every command).

    Args:
        model_path: Path to the saved model file

    Returns:
        The projection dimension used by the model
    """
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Look for the projection layer's output dimension
        # The projection layer is: nn.Linear(embedding_dim, projection_dim) -> nn.ReLU() -> nn.Linear(projection_dim, projection_dim)
        # So we want the output dimension of the final linear layer (of either tower)
        if "query_tower.projection.2.weight" in state_dict:
            # Shape is [projection_dim, projection_dim] for the final layer
            projection_dim = state_dict["query_tower.projection.2.weight"].shape[0]
            print(f"🔍 Detected projection dimension: {projection_dim}")
            return projection_dim
        else:
            print("⚠️ Could not find projection layer in model state dict, using default dimension")
            return DEFAULT_PROJ_DIM

    except Exception as e:
        print(f"⚠️ Error reading model file {model_path}: {e}")
        return DEFAULT_PROJ_DIM


def create_index_schema(
    index_name: str,
    projection_dim: int = DEFAULT_PROJ_DIM,
) -> IndexSchema:
    """Create Redis vector search index schema."""
    schema_dict = {
        "index": {
            "name": index_name,
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
        model_filename: Optional[str] = None,
        projection_dim: Optional[int] = None,
        redis_url: str = "redis://localhost:6379",
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = 6379,
        redis_pass: Optional[str] = None,
    ):
        """
        Initialize the search engine.

        Args:
            model_filename: Filename of saved model weights in /models (optional)
            projection_dim: Dimension of the model's projection layer (auto-detected if None)
            redis_url: Redis connection URL
        """
        # Initialize Redis client
        if redis_url:
            self.redis_client = Redis.from_url(redis_url)
        elif redis_host and redis_port and redis_pass:
            self.redis_client = Redis(host=redis_host, port=redis_port, password=redis_pass)
        else:
            raise ValueError("Must provide either redis URL or full host/port/pass combination")

        # Find model weights if available
        model_path = None
        model_filename = None
        if model_filename is None:
            result = find_best_model()
            if result:
                model_path, model_filename = result
        # if a specific model filename is provided, we check it exists and has appropriate extension
        elif model_filename and model_filename.endswith((".pt", ".pth")):
            path = os.path.join(MODELS_DIR, model_filename)
            if os.path.exists(path):
                model_path = path
                model_filename = model_filename
            print(f"🔍 Using custom model: {model_filename}")
        else:
            print("❌ Custom model provided but either not found to exist, or not ending with .pt/.pth")

        # init index name, using model filename if available
        self.index_name = os.path.splitext(model_filename)[0] if model_filename else DEFAULT_INDEX_NAME

        # Auto-detect projection dimension from model if not provided
        if projection_dim is None:
            if model_path:
                self.projection_dim = get_projection_dim_from_model(model_path)
            else:
                print(f"ℹ️ No model available for dimension detection, using default projection dim: {DEFAULT_PROJ_DIM}")
                self.projection_dim = DEFAULT_PROJ_DIM
        else:
            self.projection_dim = projection_dim

        # Initialize model and load weights if available
        self.model = TwoTowersModel(projection_dim=self.projection_dim)
        if model_path:
            print(f"🔍 Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        else:
            print("ℹ️ No model weights provided or found - using untrained model with random weights")
        self.model.eval()

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize search index
        self.search_index: Optional[SearchIndex] = None
        self._setup_search_index()

    def _setup_search_index(self):
        """Set up Redis search index with the appropriate schema."""
        schema = create_index_schema(
            index_name=self.index_name,
            projection_dim=self.projection_dim,
        )

        try:
            # Check if index already exists, and create it if not
            self.search_index = SearchIndex(schema=schema, redis_client=self.redis_client)
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

        # Encode all documents in batches with progress tracking
        print("Encoding documents into embeddings...")
        with torch.no_grad():
            if len(documents) <= batch_size:
                # Small batch, encode directly
                all_embeddings = self.model.encode_documents(documents)
            else:
                # Large batch, encode with progress bar
                embeddings = []
                num_batches = (len(documents) + batch_size - 1) // batch_size

                for i in tqdm(
                    range(0, len(documents), batch_size), desc="Encoding document batches", total=num_batches
                ):
                    batch_docs = documents[i : i + batch_size]
                    batch_embeddings = self.model.encode_documents(batch_docs)
                    embeddings.append(batch_embeddings.cpu())  # Move to CPU to free GPU memory

                # Concatenate all embeddings
                print("Concatenating embeddings...")
                all_embeddings = torch.cat(embeddings, dim=0)
                all_embeddings = all_embeddings.to(next(self.model.parameters()).device)

        # Convert embeddings to numpy for Redis storage
        print("Converting embeddings to numpy arrays...")
        embeddings_np = all_embeddings.cpu().numpy()

        # Prepare documents for bulk insertion
        print("Preparing documents for bulk insertion...")
        data = []
        for i, (doc_text, embedding) in tqdm(
            enumerate(zip(documents, embeddings_np)), desc="Preparing documents", total=len(documents)
        ):
            data.append(
                {
                    "id": str(i),
                    "content": doc_text,
                    "embedding": embedding.tobytes(),  # Convert to bytes for Redis storage
                }
            )

        # Bulk insert documents with progress tracking
        print("Inserting documents into Redis...")
        if self.search_index:
            # could load everything at once, but we batch it to show progress (sanity!)
            total_batches = (len(data) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting into Redis", total=total_batches):
                batch_data = data[i : i + batch_size]
                self.search_index.load(batch_data, id_field="id")
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
            normalize_vector_distance=True,  # returns distance values in [0, 1] instead of [0, 2]
        )

        # Execute search
        if not self.search_index:
            raise RuntimeError("Search index not initialized")

        print(f"Searching index '{self.index_name}' for top {top_k} results...")
        results = self.search_index.query(vector_query)

        # Format results
        formatted = []
        for result in results:
            formatted.append(
                {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "score": float(result.get("vector_distance", 0.0)),  # cosine similarity between query and given doc
                    "distance": 1.0 - float(result.get("vector_distance", 0.0)),  # convert similarity -> distance
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
    max_docs: int = -1,
    batch_size: int = 1024,
    model_filename: Optional[str] = None,
    projection_dim: Optional[int] = None,
    redis_url: str = "redis://localhost:6379",
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = 6379,
    redis_pass: Optional[str] = None,
    clear_existing: bool = True,
):
    """Build the document index from MS Marco dataset."""
    print("Building document index from *all* MS Marco datasets...")

    # Initialize search engine
    engine = DocumentSearchEngine(
        model_filename=model_filename,
        projection_dim=projection_dim,
        redis_url=redis_url,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_pass=redis_pass,
    )

    if max_docs == -1:
        # Load ALL documents from all three dataset splits
        print("Loading ALL documents from train, validation, and test splits...")
        all_unique_docs = set()

        for split in ["train", "validation", "test"]:
            try:
                dataset = MSMarcoDataset(split, max_samples=-1)  # -1 means load all
                split_docs = dataset.get_unique_passages()
                all_unique_docs.update(split_docs)
                print(f"    Added {len(split_docs)} documents from {split} split")
            except Exception as e:
                print(f"    Warning: Could not load {split} split: {e}")
                continue

        unique_docs = list(all_unique_docs)
        print(f"Total unique documents across all splits: {len(unique_docs)}")
    else:
        # Load limited documents from train split only
        print(f"Loading MS Marco train dataset (max {max_docs} documents)...")
        dataset = MSMarcoDataset("train", max_samples=max_docs)
        unique_docs = dataset.get_unique_passages()

        # ensure we are limiting to max_docs (sampling them at random for more fun!)
        if max_docs > 0 and len(unique_docs) > max_docs:
            unique_docs = random.sample(unique_docs, max_docs)

        print(f"Found {len(unique_docs)} unique documents")

    # Ingest documents
    engine.ingest_documents(unique_docs, batch_size=batch_size, clear_existing=clear_existing)

    # Print index info
    info = engine.get_index_info()
    print(f"Index info: {info}")

    return engine


def search_documents(
    query: str,
    top_k: int = 10,
    model_filename: Optional[str] = None,
    projection_dim: Optional[int] = None,
    redis_url: str = "redis://localhost:6379",
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = 6379,
    redis_pass: Optional[str] = None,
):
    """Search for documents similar to the given query using auto-selected model."""
    print(f"Searching for: '{query}'")
    print("-" * 50)

    # Initialize search engine
    engine = DocumentSearchEngine(
        model_filename=model_filename,
        projection_dim=projection_dim,
        redis_url=redis_url,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_pass=redis_pass,
    )

    # Check if index exists and has documents
    info = engine.get_index_info()
    if info.get("num_docs", 0) == 0:
        print("❌ No documents found in index. Please run with --build-index first (or simultaneously)")
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filename for saved model weights in /models dir (optional, auto-selects best model if not provided)",
    )
    parser.add_argument(
        "--dims", type=int, default=None, help="Model projection dimension (auto-detected from model if not provided)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=-1, help="Maximum documents to index (-1 for all documents from all splits)"
    )
    # note that a batch size of 1024 surfed close to the wind on RTX 4090, so may overflow on e.g. a 3090
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")

    # redis connection - expect an URL but allow host/pass for greater compatibility
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--redis-host", type=str, default=None, help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-pass", type=str, default="redispass", help="Redis pass")

    args = parser.parse_args()
    if not (args.query or args.build_index or args.index_info):
        print("ℹ️ No action specified. Use --build-index, --index-info, or provide a query.")
        parser.print_help()
        return

    # Handle different actions in appropriate order
    if args.build_index:
        print("🔨 Building document index...")
        # Build index (optionally with specific model)
        build_document_index(
            max_docs=args.max_docs,
            batch_size=args.batch_size,
            model_filename=args.model,
            projection_dim=args.dims,
            redis_url=args.redis_url,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_pass=args.redis_pass,
        )
        print("\n✅ Index built successfully!")

    if args.index_info:
        print("ℹ️ Showing index information...")
        # Show index information
        engine = DocumentSearchEngine(
            model_filename=args.model,
            projection_dim=args.dims,
            redis_url=args.redis_url,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_pass=args.redis_pass,
        )
        info = engine.get_index_info()
        print(f"\nIndex info: {info}")

    if args.query:
        print("🔍 Searching for documents related to your query...")
        # Search index (may have been pre-existing or built in the same run)
        search_documents(
            query=args.query,
            top_k=args.top_k,
            model_filename=args.model,
            projection_dim=args.dims,
            redis_url=args.redis_url,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_pass=args.redis_pass,
        )


if __name__ == "__main__":
    main()
