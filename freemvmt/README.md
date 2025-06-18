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


## Remote

As usual, we have a handy `setup.sh` script, which should be sourced after cloning the repo.

If using a next-gen Nvidia demon like RTX 5090, we have to pull torch from the nightly builds index, in which case one should `export BEAST_MODE=1` before setup.


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
  --epochs 1 \
  --margin 0.3 \
  --batch-size 64 \
  --projection-dim 32 \
  --max-samples 1000 \
  --accumulation-steps 1 \
  --num-workers 1 \
  --no-comprehensive-test \
  --no-save

# compare with heavy duty, tracked GPU job, which saves weights on finish
python main.py \
  --epochs 12 \
  --margin 0.7 \
  --batch-size 1028 \
  --projection-dim 256 \
  --max-samples 100000 \
  --accumulation-steps 4 \
  --num-workers 6

# Run hyperparameter sweep
python main.py --sweep
```

Note that providing `--max-samples -1` will result in the full dataset being traversed (~ yielding `800_000` triplets).


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

# Encode queries and documents (batching if there are many of the latter)
query_embeddings = model.encode_queries(["What is machine learning?"])
doc_embeddings = model.encode_documents_batched(["Machine learning is a subset of AI..."])

# Compute similarity
import torch
similarity = torch.cosine_similarity(query_embeddings, doc_embeddings)
```


## Sweeping

Running sweeps with wandb turns out to be super powerful! The first one I ran [uodzb69z](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/uodzb69z/workspace). In general CSVs, buffers from tmux sessions, and images of the trajectories can be found in the `sweeps/` dir. On this first sweep, I could see that:

- All the best results were for 8 epochs with 256 dimensions in the projection layer (perhaps unsurprisingly)
- However, variable batch sizes, learning rates and margins all produced good results
- Also, wandb seemed to decide very quickly that 4 accumulation steps was too many, although the same run was also the quickest (which I would expect, since this relates to GPU optimisation)

For my second run ([1ukkvqtl](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/1ukkvqtl)), I decided to control the number of samples at `50_000`, because my intuition was that big differences in the raw size of the dataset would skew my results, and I wanted to tune my hypers without worrying about that. Some thoughts:

- The immediate result that jumps out is more epochs no longer guarantees better results - 16 was certainly too many for this sample size
- The worst 2 of 5 results used a margin of `1.0`, while the best used `0.5`
- The best result was with 8 epochs, 256 batches and 128 projection dim, all of which were the lowest values in the config and which you'd expect to have the opposite correlation (although we must bear in mind the fixed sample size)
- 5 runs is actually not sufficient to read too many conclusions out of a sweep with so many variables

Another run ([yo9idsvy](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/yo9idsvy)) unseated a previous conclusion about accumulation steps being negatively correlated - in this sweep wandb determined this hyperparam to be the most importance in producing good results. This also seemed to be the first time that we could say something meaningful about the learning rate; specifically, that `1e-3` beats `1e-4`.

Also worth noting that all these sweeps are being judged against the `val_ndcg_10` metric, which is our in-epoch validation test. In the latest, *all* of them are doing very well against it (`> 0.975`), which suggests that it is *too easy*. Therefore, I decided to:

a) hike the difficulty of the validation check (e.g. by expanding the small candidate pool)
b) use a *much harder* final test as the metric against which wandb evaluates a given permutation of hyperparams

Then to run an absolute monster sweep!


## Additional resources

- ShapedAI deep dive on [the two tower model for recommendation systems](https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive)
- Wandb explainer and tutorial on [Bayesian hyperparameter optimisation](https://wandb.ai/wandb_fc/articles/reports/What-Is-Bayesian-Hyperparameter-Optimization-With-Tutorial---Vmlldzo1NDQyNzcw#what-are-hyperparameters?)
- MLOps piece on using Redis for [vector similarity search in prod](https://mlops.community/vector-similarity-search-from-basics-to-production/)
- A [whopping](https://redis.io/docs/latest/develop/get-started/vector-database/) [great](https://redis.io/docs/latest/integrate/redisvl/user_guide/getting_started/#create-a-searchindex) [boatload](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-stack/docker/) of [assorted](https://github.com/redis-developer/redis-ai-resources/) [Redis vector library documentation](https://docs.redisvl.com/en/latest/) [and](https://github.com/redis-developer/fastapi-redis-tutorial) [relevant](https://github.com/antonum/Redis-VSS-Streamlit) [examples](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/02_two_towers.ipynb)
- 
