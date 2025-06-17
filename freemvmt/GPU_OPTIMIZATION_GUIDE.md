# GPU Memory Optimization Guide

I presented Claude with my GPU utilization pattern on a remote machine via nvtop. It showed that I was making good use of compute (79%), but barely scratching the memory (14%).

When asked, it proposed the following optimisations.


## Key Optimizations Added

### 1. **Increased Default Batch Size** 
- **Before**: 128 samples per batch
- **After**: 256 samples per batch (configurable via `--batch-size`)
- **Impact**: Better GPU memory utilization, improved training efficiency

### 2. **Gradient Accumulation**
- **Feature**: `--accumulation-steps` parameter (default: 1)
- **Usage**: `python main.py --accumulation-steps 4 --batch-size 128`
- **Effective batch size**: 128 × 4 = 512 samples
- **Impact**: Simulate larger batches without exceeding memory limits

### 3. **Mixed Precision Training (AMP)**
- **Feature**: Automatic Mixed Precision using `torch.cuda.amp`
- **Default**: Enabled (disable with `--no-mixed-precision`)
- **Impact**: ~50% memory reduction, ~1.5x speed improvement
- **Memory usage**: FP16 for forward pass, FP32 for gradients

### 4. **Optimized DataLoader**
- **Multi-processing**: `--num-workers 4` (default)
- **Pinned memory**: Enabled for faster CPU→GPU transfer
- **Persistent workers**: Reduces worker initialization overhead

### 5. **Memory Monitoring**
- **Real-time tracking**: GPU memory allocation/reservation logged
- **Wandb integration**: Memory metrics tracked in experiments
- **Console output**: Memory usage shown during training


## Recommended Settings for Your 24GB GPU

### For Maximum Memory Utilization

```bash
python main.py \
  --batch-size 512 \
  --accumulation-steps 2 \
  --num-workers 6 \
  --epochs 5 \
  --max-samples 50000
```

**Effective batch size**: 1024 samples
**Expected memory usage**: ~18-20GB

### For Balanced Performance

```bash
python main.py \
  --batch-size 256 \
  --accumulation-steps 4 \
  --num-workers 4 \
  --epochs 3 \
  --max-samples 20000
```

**Effective batch size**: 1024 samples
**Expected memory usage**: ~12-16GB

### For Development/Testing

```bash
python main.py \
  --batch-size 128 \
  --accumulation-steps 2 \
  --num-workers 2 \
  --epochs 1 \
  --max-samples 1000
```

**Effective batch size**: 256 samples
**Expected memory usage**: ~6-8GB


## Technical Implementation Details

### Mixed Precision Training Loop

```python
# Automatic mixed precision context
with torch.autocast('cuda'):
    query_embeds = model.encode_queries(queries)
    positive_embeds = model.encode_documents(positives)
    negative_embeds = model.encode_documents(negatives)
    loss = criterion(query_embeds, positive_embeds, negative_embeds)

# Gradient scaling for numerical stability
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Memory-Efficient DataLoader

```python
DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True  # Reuse worker processes
)
```


## Expected Performance Improvements

### Memory Utilization:
- **Before**: ~14% (3.4GB/24GB)
- **After**: ~60-80% (15-20GB/24GB) with recommended settings

### Training Speed:
- **Mixed precision**: ~1.5x faster training
- **Larger batches**: Better GPU utilization, more stable gradients
- **DataLoader optimization**: Reduced CPU-GPU bottleneck

### Training Quality:
- **Larger effective batches**: More stable gradients, better convergence
- **Better GPU utilization**: Faster experimentation cycles


## Monitoring GPU Usage

The updated training script now logs:
- Real-time GPU memory allocation
- Memory reservation vs. allocation
- Batch processing times
- Memory metrics in wandb

Watch for these patterns:
- **Good**: Memory allocation steadily increases to 60-80% of available
- **Suboptimal**: Memory stays below 30% throughout training
- **Problem**: Out-of-memory errors (reduce batch size or accumulation steps)


## Troubleshooting

### Out of Memory (OOM):
1. Reduce `--batch-size` (e.g., 256 → 128)
2. Reduce `--accumulation-steps` (e.g., 4 → 2)
3. Reduce `--num-workers` (e.g., 6 → 2)

### Low Memory Usage:
1. Increase `--batch-size` (e.g., 256 → 512)
2. Increase `--accumulation-steps` (e.g., 1 → 4)
3. Increase `--max-samples` for larger datasets

### Slow Training

1. Ensure mixed precision is enabled (don't use `--no-mixed-precision`)
2. Increase `--num-workers` for better data pipeline
3. Check that DataLoader workers aren't bottlenecked

The optimizations should significantly improve your GPU memory utilization while maintaining or improving training speed and quality.
