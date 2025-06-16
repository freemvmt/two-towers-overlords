# Two Towers

A barebones implementation of a two-towers architecture for document retrieval using PyTorch and the MS Marco dataset.

Our 'two towers' here are the *query* tower and *document* tower.

Aiming to start with a basic full stack setup (including early deploy), using only avergae pooling for each tower. Then we will swap those NNs out for RNNs and see what improvement we get.

## Features

- **Simple Architecture**: Two separate encoders (query and document) using average pooling
- **Pre-trained Embeddings**: Uses sentence-transformers/all-MiniLM-L6-v2 for initialization
- **Triplet Loss**: Training with cosine distance-based triplet loss
- **MS Marco Integration**: Direct loading from Hugging Face datasets
- **Minimal Dependencies**: Keep it simple with just PyTorch, transformers, and datasets
- **Experiment Tracking**: Full Weights & Biases integration for logging and hyperparameter sweeps

## Files

- `model.py`: Two-towers model architecture with average pooling encoders
- `training.py`: Training loop with MS Marco data loading and triplet loss
- `main.py`: Entry point for running training
- `sweep_config.py`: Wandb hyperparameter sweep configurations

## Usage

```bash
# Install dependencies
cd freemvmt && uv sync

# Run training with default settings (includes wandb logging)
uv run python main.py

# Run training without wandb
uv run python main.py --no-wandb

# Custom training parameters
uv run python main.py --epochs 5 --batch-size 16 --learning-rate 2e-4 --max-samples 5000

# Run hyperparameter sweep
uv run python main.py --sweep

# Run custom sweep configurations
uv run python sweep_config.py --config basic --count 10
uv run python sweep_config.py --config extensive --count 20
uv run python sweep_config.py --config quick --count 5
```

## Wandb Integration

The project includes comprehensive Weights & Biases integration:

### Experiment Tracking
- **Automatic logging**: Training loss, validation NDCG@10, model parameters
- **Model watching**: Gradients and weights visualization
- **Configuration tracking**: All hyperparameters and model settings

### Hyperparameter Sweeps
- **Built-in sweeps**: Use `--sweep` flag for quick parameter exploration
- **Custom configurations**: Three predefined sweep configs (basic, extensive, quick)
- **Bayesian optimization**: Intelligent parameter search for better results

### Setup Wandb
```bash
# First time setup
wandb login

# Set your default project
export WANDB_PROJECT="two-towers-retrieval"
```

## Architecture

The model consists of:
- **Query Encoder**: Average pooling over pre-trained token embeddings
- **Document Encoder**: Identical architecture to query encoder
- **Loss Function**: Triplet loss with cosine distance (1 - cosine similarity)
- **Evaluation**: NDCG@10 for validation

## Quick Start

```python
from freemvmt.model import TwoTowersModel

# Initialize model
model = TwoTowersModel()

# Encode queries and documents
query_embeddings = model.encode_queries(["What is machine learning?"])
doc_embeddings = model.encode_documents(["Machine learning is a subset of AI..."])

# Compute similarity
import torch
similarity = torch.cosine_similarity(query_embeddings, doc_embeddings)
```

## Additional resources

- https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive
- 
