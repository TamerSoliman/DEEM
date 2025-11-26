# DEEM Pretraining Guide (Stage 1)

**Difficulty:** ⭐⭐⭐⭐⭐ Expert
**Time:** 45 minutes
**Config:** `uni_interleaved/configs/train/pretrain.yaml`

---

## Overview

Stage 1 pretraining aligns image and text representations using large-scale web data (MMC4 + LAION).

---

## Objective

Learn to:
1. Encode images into visual tokens
2. Fuse vision and language in LLM
3. Generate text from images
4. Generate images from text
5. Understand images via diffusion feedback

---

## Training Data

### MMC4 (Multimodal C4)

- **Size:** ~23,000 shards, ~10M image-text pairs
- **Format:** Interleaved documents with images
- **Download:** `python scripts/download_mmc4.py`

### LAION

- **Size:** ~1,700 shards, ~100M image-text pairs
- **Format:** Image-caption pairs
- **Download:** `python scripts/download_laion.py`

---

## Configuration

**File:** `pretrain.yaml`

```yaml
model:
  llm_model_path: "./assets/lmsys/vicuna-7b-v1.5"
  seq_len: 2048
  num_img_token: 77
  loss_img_weight: 5.0
  loss_txt_weight: 1.0
  loss_sniffer_weight: 5.0
  freeze_llm: true        # Freeze LLM, train cross-attn only
  freeze_vfm: false       # Train visual encoder
  freeze_dm: true         # Freeze SD decoder

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  num_gpus: 32
  # Global batch = 4 × 8 × 32 = 1024

  learning_rate: 1e-4     # Cross-attention
  warmup_steps: 2000
  max_steps: 100000
  save_steps: 5000
```

---

## What Gets Trained?

```python
# Trainable components:
- visual_tokenizer.sniffer        # CLIP adapter
- visual_tokenizer.perceiver      # Compression
- visual_tokenizer.proj_t/proj_i  # Projections
- mm_decoder.llama_cross_attn     # Cross-attention layers
- context_feat_proj               # Context projection
- soi_token                       # Learnable <|soi|>

# Frozen:
- mm_decoder (LLaMA layers)       # Too expensive to train
- image_decoder (SD)              # Already trained
```

---

## Loss Function

```python
total_loss = (
    loss_txt * 1.0 +        # Text generation
    loss_img * 5.0 +        # Image generation
    loss_sniffer * 5.0      # Diffusion feedback
)
```

### Text Loss

Standard language modeling on text tokens:
```python
loss_txt = CrossEntropy(predicted_tokens, ground_truth)
```

### Image Loss

Diffusion loss for image generation:
```python
noise_pred = unet(noisy_latent, timestep, context, mmfs)
loss_img = MSE(noise_pred, actual_noise)
```

### Sniffer Loss

Diffusion feedback for visual understanding:
```python
dif_embed = proj_i(vis_embed)
sd_loss = encoder(image_dec, dif_embed)
loss_sniffer = sd_loss * mask_region
```

---

## Training Script

```bash
#!/bin/bash

# pretrain.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8

torchrun --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --config uni_interleaved/configs/train/pretrain.yaml \
    --output_dir ./OUTPUT/pretrain_stage1 \
    --logging_steps 100 \
    --save_steps 5000 \
    --dataloader_num_workers 4 \
    --bf16 true \
    --tf32 true
```

---

## Monitor Training

```python
# Check logs
tail -f ./OUTPUT/pretrain_stage1/logs/train.log

# Key metrics:
# - loss_txt: Should decrease to ~2.5
# - loss_img: Should decrease to ~0.15
# - loss_sniffer: Should decrease to ~0.05
# - learning_rate: Warmup then decay
```

---

## Checkpoints

**Saved every 5000 steps:**
```
./OUTPUT/pretrain_stage1/
├── checkpoint-5000/
│   ├── pytorch_model.bin
│   ├── config.yaml
│   └── optimizer.pt
├── checkpoint-10000/
...
```

**Load checkpoint:**
```python
model = MMInterleaved(**config.model)
load_model_weights(model, "./OUTPUT/pretrain_stage1/checkpoint-50000")
```

---

## Hardware Requirements

**Minimum:**
- 8× A100 40GB GPUs
- 512GB system RAM
- 2TB SSD for data

**Recommended:**
- 32× A100 80GB GPUs
- 1TB system RAM
- 10TB NVMe SSD

---

## Training Time

**On 32× A100 80GB:**
- 100K steps: ~7 days
- Full convergence: ~10 days

**On 8× A100 40GB:**
- Reduce batch size to 2/GPU
- ~28 days

---

## Common Issues

### OOM (Out of Memory)

```yaml
# Reduce batch size
per_device_train_batch_size: 2
gradient_accumulation_steps: 16

# Enable gradient checkpointing
use_llama_gradient_checkpointing: true
```

### Slow Data Loading

```yaml
# Increase workers
dataloader_num_workers: 8

# Use SSD for data
# Copy data to /dev/shm (RAM disk)
```

### Loss Spikes

```python
# Gradient clipping
max_grad_norm: 1.0

# Reduce learning rate
learning_rate: 5e-5
```

---

## Validation

Run evaluation every 10K steps:
```python
python evaluate.py \
    --config configs/eval/vqa_eval.yaml \
    --checkpoint ./OUTPUT/pretrain_stage1/checkpoint-50000
```

**Expected performance after Stage 1:**
- VQAv2: ~62% (before fine-tuning)
- COCO Caption CIDEr: ~95 (before fine-tuning)

---

## Summary

**Duration:** ~7-10 days on 32 GPUs
**Data:** MMC4 + LAION
**Loss:** Text + Image + Sniffer
**Output:** Aligned vision-language model

**Next:** Stage 2 fine-tuning on VQA/captioning tasks

---

**Last Updated:** 2025-11-24
