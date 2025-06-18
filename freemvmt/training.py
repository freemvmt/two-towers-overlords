"""
Training script for two-towers document retrieval model using MS Marco dataset.
"""

import random
from typing import Tuple, Optional, Union

from datasets import load_dataset, IterableDataset
import torch
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import ndcg_score
import numpy as np
import wandb

from model import TwoTowersModel, TripletLoss
from tqdm import tqdm


MsMarcoDatasetItem = dict[str, Union[str, int]]
Triplet = Tuple[list[str], list[str], list[str]]


# file is littered with assertions and 'type: ignore' comments to allay Python's concerns about the HF Dataset/poor type analysis
class MSMarcoDataset(Dataset):
    """MS Marco dataset that expands each query to include all its positive passages."""

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 10_000,
        random_seed: int = 42,
    ) -> None:
        print(f"Building MS Marco {split} dataset with at most {max_samples} samples...")
        # get map-style dataset from Hugging Face
        self.dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
        assert not isinstance(self.dataset, IterableDataset)

        # randomly sample from all records in the dataset (with a seed for reproducibility)
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        dataset_indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        # expand each randomly selected record to produce all query-passage pairs
        self.data: list[MsMarcoDatasetItem] = []
        self.docs = set()
        sample_count = 0
        for idx in dataset_indices:
            item = self.dataset[idx]
            query = str(item["query"])
            query_id = int(item["query_id"])  # type: ignore

            # Add all positive passages ('documents') for this query
            try:
                for doc in item["passages"]["passage_text"]:  # type: ignore
                    if doc.strip():  # Skip empty docs
                        doc = str(doc)
                        self.data.append(
                            {
                                "query_id": query_id,
                                "query": query,
                                "positive": doc,
                            }
                        )
                        # collect up all unique passages seen simultaneously to reduce work
                        self.docs.add(doc)
                        sample_count += 1
                        # traverse full dataset if max_samples is -1
                        if max_samples == -1:
                            continue
                        # else break off when we have collected enough samples (here sample == query-doc pair)
                        if max_samples and sample_count >= max_samples:
                            break
            except (KeyError, TypeError):
                print(f"Error while building query-doc pairs for record: {item}")
                continue
            if max_samples and sample_count >= max_samples:
                break

        # save list of unique docs as list
        self.docs = list(self.docs)
        print(
            f"Built {split} dataset with {len(self.data)} query-document pairs, including {len(self.docs)} unique passages"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MsMarcoDatasetItem:
        return self.data[idx]

    def get_unique_passages(self) -> list[str]:
        """Get all unique passages from the dataset (prepared during init)."""
        return self.docs  # type: ignore


class TripletDataLoader:
    """Creates triplets for training with random negative sampling."""

    def __init__(
        self,
        dataset: MSMarcoDataset,
        batch_size: int = 128,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if (device and device.type == "cuda") else False,  # faster GPU transfer
            persistent_workers=True if num_workers > 0 else False,
        )

    def create_triplets(self, batch: list[MsMarcoDatasetItem]) -> Triplet:
        """Create triplets by randomly sampling negatives (from within the batch)."""
        queries = []
        positives = []
        negatives = []

        for i, item in enumerate(batch):
            queries.append(item["query"])
            positives.append(item["positive"])

            # Sample a negative from other items in the batch (as long as the query is different)
            while True:
                negative, query_id = self._get_random_negative(batch, i)
                if negative and query_id != item["query_id"]:
                    negatives.append(negative)
                    break

        return queries, positives, negatives

    def _get_random_negative(self, batch: list[MsMarcoDatasetItem], idx: int) -> tuple[str, int]:
        """Get a random negative sample from the batch, excluding the specified index."""
        neg_idx = random.choice([j for j in range(len(batch)) if j != idx])
        return batch[neg_idx]["positive"], batch[neg_idx]["query_id"]  # type: ignore

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


def train_epoch_optimized(
    model: TwoTowersModel,
    dataloader: TripletDataLoader,
    criterion: TripletLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    accumulation_steps: int = 2,
    log_wandb: bool = True,
) -> float:
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (queries, positives, negatives) in enumerate(dataloader):
        # Use automatic mixed precision
        with autocast(device.type):
            # Get embeddings
            query_embeds = model.encode_queries(queries)
            positive_embeds = model.encode_documents(positives)
            negative_embeds = model.encode_documents(negatives)

            # Compute triplet loss
            loss = criterion(query_embeds, positive_embeds, negative_embeds)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights after every accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Unscale loss for logging
        num_batches += 1

        # Log to wandb
        if log_wandb and batch_idx % 10 == 0:
            wandb.log(
                {
                    "batch_loss": loss.item() * accumulation_steps,
                    "batch": num_batches,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
                }
            )

        if batch_idx % (num_batches / 10) == 0:
            mem_info = (
                f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f}GB reserved"
                if torch.cuda.is_available()
                else ""
            )
            print(f"Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}, {mem_info}")

    return total_loss / num_batches


def evaluate_model(
    model: TwoTowersModel,
    dataset: MSMarcoDataset,
    sample_size: int = 200,
    min_query_groups: int = 20,
    candidate_pool_size: int = 100,
    comprehensive: bool = False,
    log_wandb: bool = False,
    wandb_prefix: str = "",
) -> Union[float, dict[str, float]]:
    """
    Evaluate model using NDCG metrics with all relevant documents per query.

    This evaluation considers all relevant documents for each query as ground truth,
    rather than just one positive document. For each query:
    1. Collect all relevant documents from the dataset
    2. Create a candidate pool with relevant + irrelevant documents
    3. Compute NDCG based on how well relevant docs are ranked

    This gives a more realistic evaluation that matches real-world retrieval scenarios.

    Args:
        model: The trained two-towers model
        dataset: The dataset to evaluate on (train/validation/test)
        sample_size: Number of samples to use from the dataset
        min_query_groups: Minimum number of unique queries to evaluate on
        candidate_pool_size: Size of candidate pool for each query (-1 => use all documents)
        comprehensive: If True, performs comprehensive evaluation with detailed output and multiple metrics
        log_wandb: Whether to log results to wandb
        wandb_prefix: Prefix for wandb logging keys (e.g., "final_test_", "val_")

    Returns:
        If comprehensive=False: float (NDCG@10 score)
        If comprehensive=True: dict containing detailed metrics
    """
    # see https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    print(f"Sampling documents for evaluation (initial sample size: {sample_size})...")
    model.eval()

    # Group by query_id to get multiple relevant documents for each query
    # We run the below loop until we have sufficient unique queries
    query_groups: dict[int, dict[str, Union[str, set[str]]]] = {}
    sample_multiplier = 1
    print(f"Grouping data by queries (target: {min_query_groups} unique queries)...")
    while len(query_groups) < min_query_groups and sample_multiplier <= 10:  # Prevent infinite loop
        sample_size_current = min(sample_size * sample_multiplier, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size_current)
        eval_data = [dataset[i] for i in indices]
        for item in eval_data:
            query_id = item["query_id"]
            if query_id not in query_groups:
                query_groups[query_id] = {  # type: ignore
                    "query": item["query"],
                    "relevant_docs": set(),
                }
            query_groups[query_id]["relevant_docs"].add(item["positive"])  # type: ignore
        sample_multiplier += 1

    # Sort query groups by no. of relevant docs collected (descending) to get most populated query groups
    query_ids = sorted(
        query_groups.keys(),
        key=lambda qid: len(query_groups[qid]["relevant_docs"]),
        reverse=True,
    )[:min_query_groups]

    # Calculate statistics about the evaluation set (needed for both modes)
    total_relevant_docs = sum(len(query_groups[qid]["relevant_docs"]) for qid in query_ids)
    avg_relevant_per_query = total_relevant_docs / len(query_ids)

    print(f"Selected {len(query_ids)} queries for comprehensive evaluation")
    print(f"  Total relevant documents across these queries: {total_relevant_docs}")
    print(f"  Average relevant docs per query: {avg_relevant_per_query:.2f}")

    # Initialize score collections based on evaluation type
    ndcg_10_scores = []
    ndcg_5_scores = [] if comprehensive else None
    ndcg_1_scores = [] if comprehensive else None

    for i, query_id in enumerate(tqdm(query_ids, desc="Evaluating queries")):
        query_data: dict[str, Union[str, set[str]]] = query_groups[query_id]
        query = query_data["query"]
        relevant_docs = query_data["relevant_docs"]

        # Create candidate pool: relevant docs + many more random irrelevant docs
        # if candidate_pool_size == -1, we use ALL docs in the dataset (for a challenge!)
        if candidate_pool_size == -1:
            all_docs = set(dataset.get_unique_passages())
            irrelevant_docs = all_docs - relevant_docs  # type: ignore
        else:
            # get all irrelevant docs from within evaluation set (ensuring no overlap with relevant docs)
            irrelevant_docs = set(
                doc
                for other_qid in query_ids
                for doc in query_groups[other_qid]["relevant_docs"]
                if other_qid != query_id
                if doc not in relevant_docs
            )
            if len(irrelevant_docs) > candidate_pool_size:
                irrelevant_docs = set(random.sample(list(irrelevant_docs), candidate_pool_size))
        # we don't make a list from union of sets here to ensure relevant docs come first in the candidate pool
        candidate_pool = list(relevant_docs) + list(irrelevant_docs)

        # Create ground-truth relevance labels (1 for relevant, 0 for irrelevant)
        true_relevance = np.array([1] * len(relevant_docs) + [0] * len(irrelevant_docs))

        # Get embeddings and compute similarities
        with torch.no_grad():
            query_embed = model.encode_queries([query])  # Shape: (1, embed_dim)  # type: ignore
            doc_embeds = model.encode_documents(candidate_pool)  # Shape: (n_docs, embed_dim)
            # Compute similarities between the query and all candidate documents
            similarities = torch.cosine_similarity(query_embed.cpu(), doc_embeds.cpu(), dim=1).numpy()

        # Calculate NDCG scores (first reshaping nparrays to 2D for sklearn compatibility)
        true_relevance_reshaped = true_relevance.reshape(1, -1)
        similarities_reshaped = similarities.reshape(1, -1)
        ndcg_10 = ndcg_score(true_relevance_reshaped, similarities_reshaped, k=10)
        ndcg_10_scores.append(ndcg_10)
        if comprehensive:
            assert ndcg_5_scores is not None and ndcg_1_scores is not None
            ndcg_5 = ndcg_score(true_relevance_reshaped, similarities_reshaped, k=5)
            ndcg_1 = ndcg_score(true_relevance_reshaped, similarities_reshaped, k=1)
            ndcg_5_scores.append(ndcg_5)
            ndcg_1_scores.append(ndcg_1)

        # Print sample results for first query (both modes)
        if i == 0:
            print("\nSample evaluation results:")
            print(f"Query: {query}")
            print(f"Number of relevant docs: {len(relevant_docs)}")
            print(f"Number of candidate docs: {len(candidate_pool)}")
            print(f"NDCG@10 score for this query: {ndcg_10:.4f}")

            # Show top-ranked documents
            ranked_indices = np.argsort(similarities)[::-1][:10]  # Top 10
            print("Top 10 ranked documents:")
            for rank, doc_idx in enumerate(ranked_indices):
                relevance = "RELEVANT" if true_relevance[doc_idx] == 1 else "irrelevant"
                doc_preview = (
                    candidate_pool[doc_idx][:100] + "..."
                    if len(candidate_pool[doc_idx]) > 100
                    else candidate_pool[doc_idx]
                )
                print(f"  {rank + 1}. [{relevance}] {doc_preview}")

    # Calculate and return results based on evaluation type
    mean_ndcg_10 = float(np.mean(ndcg_10_scores))

    if not comprehensive:
        # Simple validation mode - return just NDCG@10
        print(f"\nMean NDCG@10 across {len(query_ids)} queries: {mean_ndcg_10:.4f}")
        return mean_ndcg_10

    # Comprehensive mode - calculate detailed metrics and provide comprehensive output
    results = {
        f"{wandb_prefix}ndcg_10": mean_ndcg_10,
        f"{wandb_prefix}ndcg_5": float(np.mean(ndcg_5_scores)),  # type: ignore
        f"{wandb_prefix}ndcg_1": float(np.mean(ndcg_1_scores)),  # type: ignore
        f"{wandb_prefix}ndcg_10_std": float(np.std(ndcg_10_scores)),
        f"{wandb_prefix}queries_evaluated": len(query_ids),
        f"{wandb_prefix}total_relevant_docs": total_relevant_docs,
        f"{wandb_prefix}avg_relevant_per_query": avg_relevant_per_query,
    }

    # Print comprehensive results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print("📊 Performance Metrics:")
    print(f"   NDCG@1:  {results[f'{wandb_prefix}ndcg_1']:.4f}")
    print(f"   NDCG@5:  {results[f'{wandb_prefix}ndcg_5']:.4f}")
    print(f"   NDCG@10: {results[f'{wandb_prefix}ndcg_10']:.4f} (±{results[f'{wandb_prefix}ndcg_10_std']:.4f})")
    print("\n📈 Evaluation Set Coverage:")
    print(f"   Queries evaluated: {results[f'{wandb_prefix}queries_evaluated']}")
    print(f"   Total relevant documents: {results[f'{wandb_prefix}total_relevant_docs']}")
    print(f"   Avg relevant docs/query: {results[f'{wandb_prefix}avg_relevant_per_query']:.2f}")

    # Log to wandb if requested
    if log_wandb:
        wandb.log(results)
        print("\n✅ Comprehensive test results logged to wandb")

    return results


def run_training(
    num_epochs: int = 3,
    batch_size: int = 256,  # Increased default batch size for better GPU utilization
    learning_rate: float = 1e-4,
    max_samples: int = 10000,
    projection_dim: int = 128,
    margin: float = 0.1,
    project_name: str = "two-towers-retrieval",
    use_wandb: bool = True,
    wandb_config: Optional[dict] = None,
    accumulation_steps: int = 2,  # Gradient accumulation for effective larger batch sizes
    use_mixed_precision: bool = True,  # Enable mixed precision training
    num_workers: int = 4,  # DataLoader workers for better CPU-GPU pipeline
    run_comprehensive_test: bool = True,  # Run comprehensive test after training
) -> TwoTowersModel:
    """Main training function with GPU optimizations and wandb integration."""
    print("Initializing model and data...")

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

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
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        }
        if wandb_config:
            config.update(wandb_config)

        # Only initialize wandb if it's not already initialized (e.g., during sweeps)
        if not wandb.run:
            wandb.init(project=project_name, config=config)

    # Initialize model and move to device
    model = TwoTowersModel(projection_dim=projection_dim)
    model = model.to(device)  # Move model to GPU if available
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log model info to wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"total_parameters": sum(p.numel() for p in model.parameters())})

    # Load datasets - now using expanded dataset for maximum data utilization
    train_ds = MSMarcoDataset("train", max_samples=max_samples)
    val_ds = MSMarcoDataset("validation", max_samples=1000)

    # Create data loader with optimizations
    train_dl = TripletDataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, device=device)

    # GPU optimization settings
    effective_batch_size = batch_size * accumulation_steps
    print("Training configuration:")
    print(f"  Physical batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"  DataLoader workers: {num_workers}")

    # Initialize mixed precision scaler
    scaler = GradScaler(device.type) if use_mixed_precision and device.type == "cuda" else None

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training (we differenitiate based on presence of GPU)
        if use_mixed_precision and device.type == "cuda" and scaler is not None:
            avg_loss = train_epoch_optimized(
                model=model,
                dataloader=train_dl,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                accumulation_steps=accumulation_steps,
                log_wandb=use_wandb,
            )
        else:
            avg_loss = train_epoch(model, train_dl, criterion, optimizer, log_wandb=use_wandb)
        print(f"Average training loss: {avg_loss:.4f}")

        # in-epoch evaluation
        ndcg = evaluate_model(model, val_ds)
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

    # Run comprehensive testing on the full test dataset if enabled (after all epochs)
    print("\n" + "=" * 60)
    if run_comprehensive_test:
        print("TRAINING COMPLETED - Starting comprehensive testing")
        print("=" * 60)
        # Load test dataset for final comprehensive evaluation/testing of model
        test_dataset = MSMarcoDataset("test", max_samples=10_000)
        _ = evaluate_model(
            model=model,
            dataset=test_dataset,
            min_query_groups=500,
            candidate_pool_size=-1,  # -1 => use all documents in the dataset as candidates!
            comprehensive=True,  # Enable comprehensive mode for final eval
            log_wandb=use_wandb,
            wandb_prefix="final_test_",  # Prefix for wandb logging
        )
    else:
        print("TRAINING COMPLETED - Skipping comprehensive testing")
        print("=" * 60)

    if use_wandb:
        wandb.finish()

    return model
