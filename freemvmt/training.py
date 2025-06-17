"""
Training script for two-towers document retrieval model using MS Marco dataset.
"""

import random
from typing import Dict, List, Tuple, Optional, Union

from datasets import load_dataset, Dataset as MapDataset
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
        # get map-style dataset from Hugging Face
        self.dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
        assert isinstance(self.dataset, MapDataset)  # Runtime check

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
                        self.expanded_data.append({"query": query, "positive": str(passage), "query_id": query_id})
                        sample_count += 1

                        if max_samples and sample_count >= max_samples:
                            break
            except (KeyError, TypeError):
                # Fallback: use empty string if no passages
                self.expanded_data.append({"query": query, "positive": "", "query_id": query_id})
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
        optimizer.zero_grad()  # clear old gradients

        # Get embeddings
        query_embeds = model.encode_queries(queries)
        positive_embeds = model.encode_documents(positives)
        negative_embeds = model.encode_documents(negatives)

        # Compute triplet loss
        loss = criterion(query_embeds, positive_embeds, negative_embeds)  # compute triplet loss

        # Backward pass
        loss.backward()  # compute gradients (∇loss wrt parameters)
        optimizer.step()  # update parameters (i.e. param = param - lr * grad)

        total_loss += loss.item()
        num_batches += 1

        # Log to wandb
        if log_wandb:
            wandb.log({"batch_loss": loss.item(), "batch": num_batches})

        if num_batches % 10 == 0:
            print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def evaluate_model(
    model: TwoTowersModel,
    dataset: MSMarcoDataset,
    sample_size: int = 200,
    min_query_groups: int = 20,
    candidate_pool_size: int = 50,
) -> float:
    """
    Proper evaluation using NDCG@10 with all relevant documents per query.

    This evaluation considers all relevant documents for each query as ground truth,
    rather than just one positive document. For each query:
    1. Collect all relevant documents from the dataset
    2. Create a candidate pool with relevant + irrelevant documents
    3. Compute NDCG@10 based on how well relevant docs are ranked in top 10

    This gives a more realistic evaluation that matches real-world retrieval scenarios.
    """
    # see https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    model.eval()

    # TODO: is this quite laborious to run every epoch? should we frontload this work?
    # group by query_id to get multiple relevant documents for each query
    # we run the below loop until we have sufficient unique queries
    # (this is a random process so we may still end up with 1 document per query in some cases)
    query_groups: dict[str, dict[str, Union[str, set[str]]]] = {}
    sample_multiplier = 1
    print(f"Sampling documents for evaluation (initial sample size: {sample_size})...")
    while len(query_groups) < min_query_groups and sample_multiplier <= 10:  # Prevent infinite loop
        sample_size_current = min(sample_size * sample_multiplier, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size_current)
        eval_data = [dataset[i] for i in indices]
        for item in eval_data:
            query_id = item["query_id"]
            if query_id not in query_groups:
                query_groups[query_id] = {"query": item["query"], "relevant_docs": set()}
            query_groups[query_id]["relevant_docs"].add(item["positive"])
        sample_multiplier += 1

    # finally we sort sort query groups by no. of relevant docs collected (descending) to get most populated query IDs
    query_ids = sorted(
        query_groups.keys(),
        key=lambda qid: len(query_groups[qid]["relevant_docs"]),
        reverse=True,
    )[:min_query_groups]

    ndcg_scores = []
    print(f"Evaluating on {min_query_groups} unique queries...")
    for i, query_id in enumerate(query_ids):
        query_data = query_groups[query_id]
        query = query_data["query"]
        relevant_docs = query_data["relevant_docs"]

        # Create candidate pool: relevant docs + many more random irrelevant docs
        irrelevant_docs = set()
        for other_qid in query_ids:
            if other_qid != query_id:
                irrelevant_docs.update(query_groups[other_qid]["relevant_docs"])
        if len(irrelevant_docs) > candidate_pool_size:
            irrelevant_docs = set(random.sample(list(irrelevant_docs), candidate_pool_size))
        all_candidate_docs = list(relevant_docs | irrelevant_docs)

        # Create ground-truth relevance labels (1 for relevant, 0 for irrelevant)
        true_relevance = np.array([1] * len(relevant_docs) + [0] * len(irrelevant_docs))

        # Get embeddings and compute similarities
        with torch.no_grad():
            query_embed = model.encode_queries([query])  # Shape: (1, embed_dim)
            doc_embeds = model.encode_documents(all_candidate_docs)  # Shape: (n_docs, embed_dim)
            # Compute similarities between the query and all candidate documents
            similarities = torch.cosine_similarity(query_embed, doc_embeds, dim=1).numpy()

        # Calculate NDCG@10 - sklearn expects true_relevance and similarities as 2D arrays (one row per query)
        ndcg = ndcg_score(true_relevance.reshape(1, -1), similarities.reshape(1, -1), k=10)
        ndcg_scores.append(ndcg)

        # Print sample results for first query
        if i == 0:
            print("\nSample evaluation results:")
            print(f"Query: {query}")
            print(f"Number of relevant docs: {len(relevant_docs)}")
            print(f"Number of candidate docs: {len(all_candidate_docs)}")
            print(f"NDCG@10 score for this query: {ndcg:.4f}")

            # Show top-ranked documents
            ranked_indices = np.argsort(similarities)[::-1][:10]  # Top 10
            print("Top 10 ranked documents:")
            for rank, doc_idx in enumerate(ranked_indices):
                relevance = "RELEVANT" if true_relevance[doc_idx] == 1 else "irrelevant"
                doc_preview = (
                    all_candidate_docs[doc_idx][:100] + "..."
                    if len(all_candidate_docs[doc_idx]) > 100
                    else all_candidate_docs[doc_idx]
                )
                print(f"  {rank + 1}. [{relevance}] {doc_preview}")

    mean_ndcg = float(np.mean(ndcg_scores))
    print(f"\nMean NDCG@10 across {len(query_ids)} queries: {mean_ndcg:.4f}")
    return mean_ndcg


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
    eval_dataset = MSMarcoDataset("validation", max_samples=1000)

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
