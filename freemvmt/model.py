"""
Two-towers document retrieval model with average pooling encoders.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class AveragePoolingTower(nn.Module):
    """Simple encoder that uses pre-trained embeddings with average pooling and a trainable projection layer."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", projection_dim: int = 128):
        super().__init__()
        # we export TOKENIZERS_PARALLELISM=false to avoid the tokenizer clashing with the DL workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        # our off the shelf word embeddings clock in at 384 dims!
        self.embedding_dim = self.pretrained_model.config.hidden_size

        # Freeze the pre-trained model (we just want word embeddings for free)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Add trainable projection layer
        self.projection = nn.Sequential(
            # project word embeddings to a lower-dimensional space, then activate and run through dense layer
            nn.Linear(self.embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using average pooling over token embeddings, then project."""
        # Tokenize
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Move tokenized inputs to the same device as the model
        tokens = {k: v.to(self.pretrained_model.device) for k, v in tokens.items()}

        # Get embeddings from pretrained model
        with torch.no_grad():
            output = self.pretrained_model(**tokens)

        # Average pooling (excluding padding tokens) and normalize
        pooled_embeddings = self._mean_pooling(output, tokens["attention_mask"])
        pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)

        # Apply trainable projection
        projected = self.projection(pooled_embeddings)
        return projected

    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, output, attention_mask):
        token_embeddings = output[0]  # First element of output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TwoTowersModel(nn.Module):
    """Two-towers architecture for document retrieval."""

    def __init__(
        self,
        projection_dim: int = 128,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.query_tower = AveragePoolingTower(model_name, projection_dim)
        self.document_tower = AveragePoolingTower(model_name, projection_dim)

    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        """Encode queries into embedding vectors."""
        return self.query_tower(queries)

    def encode_documents(self, documents: list[str]) -> torch.Tensor:
        """Encode documents into embedding vectors."""
        return self.document_tower(documents)

    def encode_documents_batched(
        self,
        documents: list[str],
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Encode documents in batches to avoid memory issues with eval of large document collections."""
        if len(documents) <= batch_size:
            return self.encode_documents(documents)

        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_embeddings = self.encode_documents(batch_docs)
            embeddings.append(batch_embeddings.cpu())  # Move to CPU to free GPU memory

        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        return all_embeddings.to(next(self.parameters()).device)

    def forward(self, queries: list[str], documents: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning query and document embeddings."""
        query_embeddings = self.encode_queries(queries)
        # the forward pass is subject to training batch size control, so no need to use batching method
        doc_embeddings = self.encode_documents(documents)
        return query_embeddings, doc_embeddings


# NB. could swap this out for torch.nn.TripletMarginWithDistanceLoss
class TripletLoss(nn.Module):
    """Triplet loss with cosine distance."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance (1 - cosine similarity)."""
        cosine_sim = F.cosine_similarity(x, y, dim=1)
        return 1 - cosine_sim

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss using cosine distance."""
        # the 'anchor' is the query vector/embedding
        pos_dist = self.cosine_distance(anchor, positive)
        neg_dist = self.cosine_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        # we return the mean loss across all triplets in the batch
        return loss.mean()
