# Two Towers

A barebones implementation of a two-towers architecture for document retrieval using PyTorch and the MS Marco dataset.

Our 'two towers' here are the *query* tower and *document* tower.

Aiming to start with a basic full stack setup (including early deploy), using only avergae pooling for each tower. Then we will swap those NNs out for RNNs and see what improvement we get.

## TODO

- [ ] Build frontend for running queries and seeing results
- [ ] Swap average pooling tower out for an RNN and compare like for like (test as a hyperparam?)
- [ ] Try using the v2.1 dataset


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
  --max-samples 10000 \
  --batch-size 256 \
  --epochs 2 \
  --learning-rate 0.001 \
  --projection-dim 64 \
  --margin 0.3 \
  --num-workers 2 \
  --no-wandb \
  --no-save \
  --no-comprehensive-test \
  --no-mixed-precision

# compare with heavy duty, tracked GPU job, which saves weights on finish
python main.py \
  --max-samples 200000 \
  --batch-size 2048 \
  --epochs 15 \
  --learning-rate 0.0001 \
  --projection-dim 512 \
  --margin 0.3 \
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

Then I ran an absolute monster sweep! [8x5w5jxn](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/8x5w5jxn/workspace) suggests as reliable hyperparams to try against full runs, we should stick to a formula along the lines of...

- *A projection dimension of `512` (in this experiment, more was better, but jumping to `1024` seems drastic?)
- A margin of `0.5` (`0.4`, which we haven't tested, could be good)
- Only `2` accumulation steps - the results are consistently mixed on this, so let's just freeze it
- Batch size of `2048` (or `1024` for a less stonky GPU)
- *Learning rate of `1e-4` (`1e-2` is too high and should be avoided, but we should potentially consider `1e-5`)
- We likely need `15` epochs or more (especially for full dataset)

i.e. if we wanted to use the *flowing* approach on the full dataset, we might run:

```
python main.py \
  --max-samples -1 \
  --batch-size 2048 \
  --epochs 15 \
  --learning-rate 0.0001 \
  --projection-dim 512 \
  --margin 0.5 \
  --num-workers 6
```

These points speak to `flowing_sweep`, which was the best run by a margin (still only just peaking about `0.5` for our eval metric). However, `iconic_sweep` is also of interest! It performed very well on `200_000` samples (i.e. harder), with a proj. dim of `128` and only `9` epochs of training! This might be explained by the faster learning rate of `1e-3`. Interestingly, it also used a margin of `0.3` (whereas the other top results used `0.5`) and managed to make the best use of GPU memory (with an effective batch size of `2048 * 4 = 8192`).

This suggests that if we want to do quicker runs to test new hyperparams, or the effect of replacing the average pooling tower with an RNN, we could use the *iconic* approach:

```
python main.py \
  --max-samples -1 \
  --batch-size 2048 \
  --epochs 9 \
  --learning-rate 0.001 \
  --projection-dim 128 \
  --margin 0.3 \
  --num-workers 6
```

Finally, I did one final sweep ([1zh189kd](https://wandb.ai/freemvmt-london/two-towers-retrieval/sweeps/1zh189kd)) to home in on margin and learning rate, controlling all other params (e.g. 15 epochs, 512 dims) and using `grid` mode to cover all options. The clear result was that learning rate was super important, and should certainly be the higher value of `1e-4` - `1e-5` is simply too slow. A margin of `0.3` also came out on top, although it seems to be much less important than LR.

This suggests a near-optimal setup for a full run (with avg. pooling towers, that is), might be:

```
python main.py \
  --max-samples -1 \
  --batch-size 2048 \
  --epochs 15 \
  --learning-rate 0.0001 \
  --projection-dim 512 \
  --margin 0.3 \
  --num-workers 6
```


## Redis vector search

The `search.py` module implements a Redis-based vector search system for the Two Towers document retrieval model described above. It provides efficient/approximate nearest neighbor (ANN) search using pre-computed document embeddings.

### Features

- **HNSW Vector Index**: Uses Hierarchical Navigable Small World algorithm for fast similarity search
- **Multi-Dataset Support**: Indexes all documents from train, validation, and test splits by default
- **Flexible Batching**: Configurable batch sizes for memory-efficient processing  
- **Cosine Similarity**: Optimized for semantic matching of a given query to k documents
- **Production Ready**: Redis backend for scalable deployment

### Quick Start

For dev, first make sure you have the `redis-stack` [Docker container](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-stack/docker/) running on local. This includes an *insights* GUI `localhost:8001`.

```sh
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Index ALL documents from ALL dataset splits (default). This will look in `models/` for the weights produced with the greatest number of epochs (or if it finds none, will just use random embeddings). Note that you can override this selection process by naming a model state file like `model/weights.py`.

```bash
python search.py --build-index
```

Check index status after building:

```bash
python search.py --index-info
```

Then you can run a search against any query you like, and supply an argument to return only so many results:

```bash
python search.py "best coffee in Old Street" --top-k 5
```

You can also index a limited number of documents using a specific model for dev/debug runs:

```bash
python search.py --build-index --max-docs 1000 --model custom_weights.pt
```

Note that if you built your index with any model other than the one returned by `find_best_model`, you will need to specify it when you search a query, to ensure that the query is encoded by the same model as the documents which are being searched against.

```bash
python search.py "deep neural networks" --model custom_weights.pt
```

And finally, you can do it all in one go!

```bash
python search.py --build-index --model custom_weights.pt --index-info --top-k 20 "what is a hummingbird moth" 
```

NB. If for some reason the wrong projection dimension value is being read out of the model state, you can override with a flag, e.g. `--dims 512`.

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-docs` | `-1` | Number of documents to index (-1 = all) |
| `--batch-size` | `1024` | Batch size for document processing |
| `--top-k` | `10` | Number of search results to return |
| `--projection-dim` | `None` | Model embedding dimension |
| `--model` | `None` | Name of weights file in models/ dir (ending `.pt`/`.pth`) |
| `--redis-url` | `redis://localhost:6379` | Redis connection URL |

### Index Schema

The Redis index uses the following schema:

```python
{
    "index": {
        "name": index_name,  # takes name from model, or else is called 'default_index'
        "prefix": "doc:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "id", "type": "tag"},
        {"name": "content", "type": "text"},
        {
            "name": "embedding", 
            "type": "vector",
            "attrs": {
                "dims": 128,  # also pulled from model, or else set at 128
                "algorithm": "hnsw",
                "distance_metric": "cosine",
            }
        }
    ]
}
```

### Architecture

```
Query Text → Query Tower → Query Embedding
                                ↓
                         Redis Vector Search (HNSW)
                                ↓
                    Top-K Similar Documents ← Document Embeddings ← Document Tower ← Document Texts
```

### Performance Notes

- **Memory Usage**: ~4 bytes per dimension per document for embeddings
- **Index Size**: For 1M documents with 128-dim embeddings: ~500MB
- **Search Speed**: Sub-millisecond search times with HNSW
- **Throughput**: 1000+ queries/second depending on hardware


## Additional resources

- ShapedAI deep dive on [the two tower model for recommendation systems](https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive)
- Wandb explainer and tutorial on [Bayesian hyperparameter optimisation](https://wandb.ai/wandb_fc/articles/reports/What-Is-Bayesian-Hyperparameter-Optimization-With-Tutorial---Vmlldzo1NDQyNzcw#what-are-hyperparameters?)
- MLOps piece on using Redis for [vector similarity search in prod](https://mlops.community/vector-similarity-search-from-basics-to-production/)
- A [whopping](https://redis.io/docs/latest/develop/get-started/vector-database/) [great](https://redis.io/docs/latest/integrate/redisvl/user_guide/getting_started/#create-a-searchindex) [boatload](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-stack/docker/) of [assorted](https://github.com/redis-developer/redis-ai-resources/) [Redis vector library documentation](https://docs.redisvl.com/en/latest/) [and](https://github.com/redis-developer/fastapi-redis-tutorial) [relevant](https://github.com/antonum/Redis-VSS-Streamlit) [examples](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/02_two_towers.ipynb)
