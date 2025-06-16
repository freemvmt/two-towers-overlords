"""
Training script for two-towers document retrieval model using MS Marco dataset.
"""

import random
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset, Dataset as HFDataset
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import ndcg_score
import numpy as np
import wandb

from model import TwoTowersModel, TripletLoss


class MSMarcoDataset(Dataset):
    """MS Marco dataset that expands each query to include all its positive passages."""
    
    def __init__(self, split: str = "train", max_samples: int = 10000):
        print(f"Loading MS Marco {split} dataset...")
        self.dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
        assert isinstance(self.dataset, HFDataset)  # Runtime check
        
        # Expand dataset to include all query-passage pairs
        self.expanded_data = []
        sample_count = 0
        
        for item in self.dataset:
            query = str(item["query"])
            query_id = str(item["query_id"])
            
            # Add all positive passages for this query
            try:
                passage_texts = item["passages"]["passage_text"]
                for passage in passage_texts:
                    if passage.strip():  # Skip empty passages
                        self.expanded_data.append({
                            "query": query,
                            "positive": str(passage),
                            "query_id": query_id
                        })
                        sample_count += 1
                        
                        if max_samples and sample_count >= max_samples:
                            break
            except (KeyError, TypeError):
                # Fallback: use empty string if no passages
                self.expanded_data.append({
                    "query": query,
                    "positive": "",
                    "query_id": query_id
                })
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
                
        print(f"Expanded to {len(self.expanded_data)} query-passage pairs")
    
    def __len__(self) -> int:
        return len(self.expanded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.expanded_data[idx]


class TripletDataLoader:
    """Creates triplets for training with random negative sampling."""

    def __init__(self, dataset: MSMarcoDataset, batch_size: int = 128):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def create_triplets(self, batch: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
        """Create triplets by randomly sampling negatives from the batch."""
        queries = []
        positives = []
        negatives = []

        for i, item in enumerate(batch):
            queries.append(item["query"])
            positives.append(item["positive"])

            # Sample a negative from other items in the batch (as long as the query is different)
            while True:
                negative, query_id = self._get_random_negative(batch, i)
                if negative and query_id != int(item["query_id"]):
                    negatives.append(negative)
                    break

        return queries, positives, negatives
  
    def _get_random_negative(self, batch: List[Dict[str, str]], idx: int) -> tuple[str, int]:
        """Get a random negative sample from the batch, excluding the specified index."""
        neg_idx = random.choice([j for j in range(len(batch)) if j != idx])
        return batch[neg_idx]["positive"], int(batch[neg_idx]["query_id"])

    def __iter__(self):
        for batch in self.dataloader:
            # Convert batch to list of dicts
            batch_list = []
            for i in range(len(batch["query"])):
                batch_list.append(
                    {"query": batch["query"][i], "positive": batch["positive"][i], "query_id": batch["query_id"][i]}
                )

            yield self.create_triplets(batch_list)


def train_epoch(
    model: TwoTowersModel,
    dataloader: TripletDataLoader,
    criterion: TripletLoss,
    optimizer: torch.optim.Optimizer,
    log_wandb: bool = True,
) -> float:
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0
    for queries, positives, negatives in dataloader:
        optimizer.zero_grad()

        # Get embeddings
        query_embeddings = model.encode_queries(queries)
        positive_embeddings = model.encode_documents(positives)
        negative_embeddings = model.encode_documents(negatives)

        # Compute triplet loss
        loss = criterion(query_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log to wandb
        if log_wandb:
            wandb.log({"batch_loss": loss.item(), "batch": num_batches})

        if num_batches % 10 == 0:
            print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def evaluate_model(model: TwoTowersModel, dataset: MSMarcoDataset, sample_size: int = 100) -> float:
    """Simple evaluation using NDCG@10."""
    model.eval()

    # Sample a subset for evaluation
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    eval_data = [dataset[i] for i in indices]

    queries = [item["query"] for item in eval_data]
    documents = [item["positive"] for item in eval_data]

    with torch.no_grad():
        query_embeddings = model.encode_queries(queries)
        doc_embeddings = model.encode_documents(documents)

        # Compute similarities
        similarities = torch.cosine_similarity(
            query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2
        ).numpy()

    # Calculate NDCG@10 (simplified - assumes perfect relevance for matched pairs)
    ndcg_scores = []
    for i in range(len(queries)):
        # Ground truth: the i-th document is relevant to the i-th query
        true_relevance = np.zeros(len(documents))
        true_relevance[i] = 1

        # Predicted relevance scores
        pred_scores = similarities[i]

        # Calculate NDCG@10
        ndcg = ndcg_score([true_relevance], [pred_scores], k=10)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def run_training(
    num_epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    max_samples: int = 1000,
    projection_dim: int = 128,
    margin: float = 0.1,
    project_name: str = "two-towers-retrieval",
    use_wandb: bool = True,
    wandb_config: Optional[Dict] = None,
) -> TwoTowersModel:
    """Main training function with wandb integration."""
    print("Initializing model and data...")

    # Initialize wandb if enabled
    if use_wandb:
        config = {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "loss_function": "triplet_loss",
            "distance_metric": "cosine",
            "margin": margin,
            "dataset_type": "ms_marco_all_passages",  # Using all passages by default
        }
        if wandb_config:
            config.update(wandb_config)

        wandb.init(project=project_name, config=config)

    # Initialize model
    model = TwoTowersModel(projection_dim=projection_dim)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log model info to wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"total_parameters": sum(p.numel() for p in model.parameters())})

    # Load datasets - now using expanded dataset for maximum data utilization
    train_dataset = MSMarcoDataset("train", max_samples=max_samples)
    eval_dataset = MSMarcoDataset("validation", max_samples=100)

    # Create data loader
    train_loader = TripletDataLoader(train_dataset, batch_size=batch_size)

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, log_wandb=use_wandb)
        print(f"Average training loss: {avg_loss:.4f}")

        # Evaluation
        ndcg = evaluate_model(model, eval_dataset)
        print(f"Validation NDCG@10: {ndcg:.4f}")

        # Log epoch metrics to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_loss,
                    "val_ndcg_10": ndcg,
                }
            )

    if use_wandb:
        wandb.finish()

    return model
