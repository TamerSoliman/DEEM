# Multi-GPU Training

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

DEEM uses PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training with gradient synchronization and model sharding.

---

## Basic Setup

**File:** `train.py:45-78`

### Launch Command

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 \
    train.py \
    --config configs/train/pretrain_stage1.yaml \
    --output_dir ./OUTPUT/pretrain

# Multi-node (2 nodes, 8 GPUs each = 16 total)
# Node 0:
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py --config ...

# Node 1:
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py --config ...
```

---

## Initialization

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize distributed training"""
    # Environment variables set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # NCCL for GPU communication
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(local_rank)

    return local_rank, world_size, rank

# Usage
local_rank, world_size, rank = setup_distributed()
```

---

## Model Wrapping

```python
# Create model on local GPU
model = MMInterleaved(**config.model)
model = model.to(local_rank)

# Wrap with DDP
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # Set True if model has unused params
    broadcast_buffers=True,
    gradient_as_bucket_view=True  # Memory optimization
)
```

---

## Data Loading

**File:** `uni_interleaved/custom_datasets/utils/build.py:220-250`

### Distributed Sampler

```python
from torch.utils.data.distributed import DistributedSampler

# Create dataset
dataset = build_train_dataset(config.data)

# Distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=config.per_device_train_batch_size,
    sampler=sampler,  # Use sampler instead of shuffle
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
```

### Epoch Synchronization

```python
# Set epoch for sampler (ensures different shuffle each epoch)
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)

    for batch in dataloader:
        # Training step
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
```

---

## Gradient Synchronization

### Automatic Synchronization

```python
# DDP automatically synchronizes gradients across GPUs
for batch in dataloader:
    outputs = model(batch)
    loss = outputs["loss"]

    optimizer.zero_grad()
    loss.backward()  # Gradients sync here (all-reduce)
    optimizer.step()
```

### Gradient Accumulation

```python
gradient_accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # Disable gradient sync until accumulation is complete
    with model.no_sync() if (i + 1) % gradient_accumulation_steps != 0 else nullcontext():
        outputs = model(batch)
        loss = outputs["loss"] / gradient_accumulation_steps

        loss.backward()

    # Sync and update every N steps
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Loss Aggregation

```python
# Compute loss on each GPU
loss_local = outputs["loss"]

# Average across all GPUs
loss_global = loss_local.detach().clone()
dist.all_reduce(loss_global, op=dist.ReduceOp.AVG)

# Only rank 0 logs
if rank == 0:
    print(f"Global loss: {loss_global.item():.4f}")
```

---

## Saving and Loading

### Save Only on Rank 0

```python
if rank == 0:
    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.module.state_dict(),  # .module to unwrap DDP
        "optimizer": optimizer.state_dict(),
        "config": config
    }, f"checkpoint-{step}.pth")

# Wait for rank 0 to finish saving
dist.barrier()
```

### Load on All Ranks

```python
# Load checkpoint
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=f"cuda:{local_rank}"
)

# Load state dict
model.module.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

# Synchronize
dist.barrier()
```

---

## Monitoring with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# Only rank 0 writes to TensorBoard
if rank == 0:
    writer = SummaryWriter(log_dir="./logs")

# Training loop
for step, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs["loss"]

    # ... training step ...

    # Log on rank 0
    if rank == 0 and step % 10 == 0:
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
```

---

## Memory Optimization

### 1. Gradient Checkpointing

```python
# Recompute activations during backward pass
model.mm_decoder.gradient_checkpointing_enable()

# Memory savings: ~30-40%
# Speed cost: ~20% slower
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = outputs["loss"]

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. CPU Offloading

```python
# Offload optimizer states to CPU
from deepspeed.ops.adam import DeepSpeedCPUAdam

optimizer = DeepSpeedCPUAdam(
    model.parameters(),
    lr=1e-4,
    adamw_mode=True
)
```

---

## Performance Tuning

### Measure Throughput

```python
import time

# Warm-up
for _ in range(10):
    outputs = model(batch)

# Measure
start = time.time()
for i in range(100):
    outputs = model(batch)
torch.cuda.synchronize()
end = time.time()

throughput = 100 * batch_size / (end - start)
if rank == 0:
    print(f"Throughput: {throughput:.2f} samples/sec")
```

### Expected Throughput

| GPUs | Batch/GPU | Total Batch | Throughput | Speedup |
|------|-----------|-------------|------------|---------|
| 1 | 16 | 16 | 12 samples/s | 1.0× |
| 2 | 16 | 32 | 23 samples/s | 1.9× |
| 4 | 16 | 64 | 45 samples/s | 3.8× |
| 8 | 16 | 128 | 85 samples/s | 7.1× |

**Scaling efficiency:** ~90%

---

## Common Issues

### 1. Hanging on Initialization

```python
# Cause: Incorrect NCCL configuration
# Solution: Set environment variables

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1  # Disable P2P if issues
```

### 2. Out of Memory

```python
# Reduce batch size per GPU
per_device_train_batch_size: 8  # Instead of 16

# Enable gradient accumulation
gradient_accumulation_steps: 4

# Effective batch size = 8 * 4 * 8 GPUs = 256
```

### 3. Unused Parameters

```python
# Error: "RuntimeError: Expected to have finished reduction..."
# Cause: Some parameters don't receive gradients

# Solution 1: Set find_unused_parameters=True
model = DDP(model, find_unused_parameters=True)

# Solution 2: Freeze unused parameters
for param in model.visual_tokenizer.parameters():
    param.requires_grad = False
```

---

## DeepSpeed Integration

```yaml
# deepspeed_config.json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

```bash
# Launch with DeepSpeed
deepspeed --num_gpus=8 train.py \
    --config configs/train/pretrain_stage1.yaml \
    --deepspeed deepspeed_config.json
```

---

## Summary

**Setup:** torchrun + DistributedDataParallel
**Scaling:** ~90% efficiency on 8 GPUs
**Memory:** Use gradient checkpointing + mixed precision
**Monitoring:** TensorBoard on rank 0 only

---

**Last Updated:** 2025-11-24
