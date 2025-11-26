# WebDataset Streaming

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 25 minutes

---

## Overview

WebDataset enables efficient streaming of large-scale datasets (e.g., MMC4's 103M samples) without downloading everything locally.

---

## Installation

```bash
pip install webdataset==0.2.5
```

---

## Dataset Format

**File:** `uni_interleaved/custom_datasets/train/mmc4_wds.py`

### Shard Structure

```
mmc4/
├── train-0000.tar
├── train-0001.tar
├── train-0002.tar
└── ...

# Each tar contains:
000000.jpg
000000.json
000001.jpg
000001.json
...
```

### JSON Format

```json
{
  "images": ["000000.jpg", "000001.jpg"],
  "texts": ["A cat sitting on a mat.", "The cat is orange."],
  "image_info": [
    {"face_detections": null},
    {"face_detections": null}
  ]
}
```

---

## Implementation

```python
import webdataset as wds

class MMC4WebDataset:
    def __init__(self, data_dir, num_samples, transform):
        self.transform = transform

        # Create WebDataset pipeline
        self.dataset = (
            wds.WebDataset(f"{data_dir}/train-{{0000..1023}}.tar")
            .shuffle(1000)                # Shuffle buffer
            .decode("pil")                # Decode images
            .to_tuple("json", "jpg")      # Extract fields
            .map(self.process_sample)     # Custom processing
            .with_length(num_samples)     # Set length for DataLoader
        )

    def process_sample(self, json_data, image):
        """Process single sample"""
        # Apply dual-resolution transform
        image_enc, image_dec = self.transform(image)

        # Extract text
        text = json_data["texts"][0] if json_data["texts"] else ""

        return {
            "image_tensors": image_enc,
            "image_tensors_dec": image_dec,
            "text": text
        }
```

---

## Advanced Pipeline

```python
# Full pipeline with filtering and augmentation
dataset = (
    wds.WebDataset(urls, shardshuffle=True)
    .shuffle(1000)
    .decode("pil", handler=wds.warn_and_continue)  # Skip corrupted images
    .to_tuple("json", "jpg")
    .map(lambda x: process_sample(x, transform))
    .select(lambda x: x is not None)  # Filter invalid samples
    .batched(32, collation_fn=custom_collator)  # Batch samples
)
```

---

## DataLoader Integration

```python
from torch.utils.data import DataLoader

# WebDataset doesn't need shuffling in DataLoader
# (already shuffled in pipeline)
dataloader = DataLoader(
    dataset,
    batch_size=None,      # Batching done in pipeline
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

# Training loop
for batch in dataloader:
    images = batch["image_tensors"]
    texts = batch["texts"]
    # Train...
```

---

## Multi-Image Interleaved Format

**File:** `uni_interleaved/custom_datasets/train/mmc4_wds.py:96`

```python
def __getitem__(self, idx):
    json_data, *images = self.dataset[idx]

    # Multiple images in sequence
    num_images = len(images)

    # Transform all images
    images_enc = []
    images_dec = []
    for img in images:
        img_enc, img_dec = self.transform(img)
        images_enc.append(img_enc)
        images_dec.append(img_dec)

    # Interleave text and images
    # "<|soi|> <|image|>×77 text1 <|soi|> <|image|>×77 text2"
    sequence = ""
    for i, text in enumerate(json_data["texts"]):
        if i < num_images:
            sequence += f"<|soi|>{'<|image|>' * 77}"
        sequence += f" {text}"

    return {
        "image_tensors": torch.stack(images_enc),      # (N, 3, 224, 224)
        "image_tensors_dec": torch.stack(images_dec),  # (N, 3, 512, 512)
        "text": sequence,
        "num_images": num_images
    }
```

---

## Sharding for Multi-GPU

```python
# Automatic sharding across workers
dataset = (
    wds.WebDataset(urls)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("json", "jpg")
    .map(process)
)

# DataLoader with multiple workers
# Worker 0 reads shards 0, 4, 8, ...
# Worker 1 reads shards 1, 5, 9, ...
# Worker 2 reads shards 2, 6, 10, ...
# Worker 3 reads shards 3, 7, 11, ...
dataloader = DataLoader(dataset, num_workers=4, batch_size=None)
```

---

## Performance Optimization

### 1. Shuffle Buffer Size

```python
# Small buffer (faster, less random)
dataset.shuffle(100)

# Large buffer (slower, more random)
dataset.shuffle(10000)

# Recommended for large datasets
dataset.shuffle(1000)
```

### 2. Prefetching

```python
dataloader = DataLoader(
    dataset,
    num_workers=4,
    prefetch_factor=2,  # Prefetch 2 batches per worker
    pin_memory=True     # Faster GPU transfer
)
```

### 3. Shard Shuffling

```python
# Shuffle shards before reading
dataset = wds.WebDataset(urls, shardshuffle=True)
```

---

## Error Handling

```python
# Skip corrupted samples
dataset = (
    wds.WebDataset(urls)
    .decode("pil", handler=wds.warn_and_continue)  # Skip decode errors
    .to_tuple("json", "jpg", handler=wds.warn_and_continue)
    .map(process, handler=wds.warn_and_continue)  # Skip processing errors
)
```

---

## Monitoring

```python
# Track samples processed
from tqdm import tqdm

for i, batch in enumerate(tqdm(dataloader)):
    # Process batch
    if i % 100 == 0:
        print(f"Processed {i * batch_size} samples")
```

---

## Common Issues

### 1. Length Unknown

```python
# WebDataset doesn't know length by default
# Manually specify:
dataset.with_length(103_000_000)  # MMC4 size
```

### 2. Reproducibility

```python
# Set seed for shuffling
import random
random.seed(42)

dataset = wds.WebDataset(urls, shardshuffle=True)
```

### 3. Memory Leaks

```python
# Close DataLoader workers properly
for epoch in range(num_epochs):
    dataloader = DataLoader(dataset, ...)
    for batch in dataloader:
        train(batch)
    del dataloader  # Clean up workers
```

---

## Summary

**Format:** .tar shards with image-text pairs
**Benefits:** Stream 100M+ samples, no local storage
**Performance:** 1000-sample shuffle buffer, 4 workers

---

**Last Updated:** 2025-11-24
