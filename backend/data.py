import random
from typing import Tuple, Optional, Union

from datasets import load_dataset, IterableDataset
from torch import Generator, randperm, device
from torch.utils.data import DataLoader, Dataset

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
        samples_txt = f"at most {max_samples}" if max_samples > 0 else "all"
        print(f"Building MS Marco {split} dataset with {samples_txt} samples...")
        # validate max_samples
        if max_samples < -1 or max_samples == 0:
            raise ValueError(
                "max_samples must be -1 (use full dataset) or > 0 (limit to that many samples)"
            )

        # get map-style dataset from Hugging Face
        self.dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
        assert not isinstance(self.dataset, IterableDataset)

        # randomly sample from all records in the dataset (with a seed for reproducibility)
        generator = Generator()
        generator.manual_seed(random_seed)
        dataset_indices = randperm(len(self.dataset), generator=generator).tolist()

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
                        # traverse full dataset if max_samples == -1 (and therefore never satisfies max_samples > 0)
                        # else break off when we have collected enough samples (here sample == query-doc pair)
                        if max_samples > 0 and sample_count >= max_samples:
                            break
            except (KeyError, TypeError):
                print(f"Error while building query-doc pairs for record: {item}")
                continue
            if max_samples > 0 and sample_count >= max_samples:
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
        batch_size: int = 1024,
        num_workers: int = 4,
        device: Optional[device] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
            if (device and device.type == "cuda")
            else False,  # faster GPU transfer
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

    def _get_random_negative(
        self, batch: list[MsMarcoDatasetItem], idx: int
    ) -> tuple[str, int]:
        """Get a random negative sample from the batch, excluding the specified index."""
        neg_idx = random.choice([j for j in range(len(batch)) if j != idx])
        return batch[neg_idx]["positive"], batch[neg_idx]["query_id"]  # type: ignore

    def __iter__(self):
        for batch in self.dataloader:
            # Convert batch to list of dicts
            batch_list = []
            for i in range(len(batch["query"])):
                batch_list.append(
                    {
                        "query": batch["query"][i],
                        "positive": batch["positive"][i],
                        "query_id": batch["query_id"][i],
                    }
                )

            yield self.create_triplets(batch_list)
