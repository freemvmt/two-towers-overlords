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
python main.py

# Run training without wandb
python main.py --no-wandb

# Custom training parameters (args here are good for dev/debug on CPU)
python main.py \
  --no-wandb \
  --epochs 2 \
  --margin 0.1 \
  --batch-size 128 \
  --projection-dim 64 \
  --max-samples 1000 \
  --accumulation-steps 1 \
  --num-workers 1

# compare with heavy duty, tracked GPU job
python main.py \
  --epochs 3 \
  --margin 0.1 \
  --batch-size 256 \
  --projection-dim 128 \
  --max-samples 10000 \
  --accumulation-steps 2 \
  --num-workers 4

# Run hyperparameter sweep
python main.py --sweep
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

### Making sense of all this

A *forward* pass does something like this:

```
query/doc → tokenizer → word embeddings → pooling → projection → vector embeddings
(query_embeds, positive_embeds, negative_embeds) → triplet_loss → loss (scalar)
```

Then on the *backward* pass (`loss.backward()`), the gradient of each parameter (wrt our calculated triplet loss) in the projection layer in each tower is calculated, and is then used to determine whether to increase or decrease the corresponding weight (`optimizer.step()`). The amount by which we make this adjustment is determined by the learning rate. So the same loss value is propagated back through each tower, but their weights are independent!

A note on loss: instead of *minimising* cosine difference, we could *maximise* cosine similarity. However, PyTorch and its optimisers generally assume we want to minimise loss, so this seems the more intuitive setup.

#### Loss landscapes

Claude had a helpful little description of how to think about this process.

```
Loss Landscape:
     /\    /\
    /  \  /  \     ← High loss (bad)
   /    \/    \
  /            \
 /              \  ← Low loss (good)
```

- Loss: Your current height on the hill
- Gradient: The slope of the hill at your current position
- Negative gradient: Points downhill (direction to reduce loss)

### Eval

We use the 'Normalized Discounted Cumulative Gain' technique to evaluate our model at each epoch against the validation dataset.

We borrow an [sklearn method](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) in our implementation. See [this article](https://www.evidentlyai.com/ranking-metrics/ndcg-metric) for a deep dive.

To put it more plainly, it essentially asks, "if I give you a query and a bunch of documents (including the 10 most relevant docs, as per the data), how many can you rank in the top 10?"

Claude suggests the following as a guide to understand the result in each case:

```
untrained_model = 0.1-0.3    # Random embeddings
decent_model = 0.4-0.6       # Basic trained model  
good_model = 0.6-0.8         # Well-trained model
excellent_model = 0.8-0.9    # SOTA approaches
perfect_model = 1.0          # Theoretical maximum (never achieved in practice)
```


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


## Sweeping

Running sweeps with wandb turns out to be super powerful! The first one we ran [uodzb69z](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/uodzb69z/workspace?nw=nwuserfreemvmt). In general CSVs, buffers from tmux sessions, and images of the trajectories can be found in the `sweeps/` dir.

Conclusions we drew from conducting hyperparameter sweep experiments:
- 



## Additional resources

- https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive
