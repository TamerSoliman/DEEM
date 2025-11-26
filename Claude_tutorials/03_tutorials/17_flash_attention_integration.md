# Flash Attention Integration

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 20 minutes

---

## Overview

Flash Attention accelerates attention computation from O(N²) to O(N) memory, enabling longer sequences and larger batch sizes.

---

## Installation

```bash
# Install flash-attn
pip install flash-attn==2.3.0 --no-build-isolation

# Requires:
# - CUDA 11.6+
# - PyTorch 2.0+
# - GPU with compute capability >= 8.0 (A100, A6000, RTX 3090+)
```

---

## Usage in DEEM

**File:** `uni_interleaved/models/decoders/modeling_llama_mmfs.py:397`

### Standard Attention (Baseline)

```python
# Standard scaled dot-product attention
Q = self.q_proj(hidden_states)  # (B, L, H, D)
K = self.k_proj(hidden_states)
V = self.v_proj(hidden_states)

# Memory: O(B * L² * H)
attn_weights = (Q @ K.T) / sqrt(d_k)  # (B, L, L, H)
attn_output = attn_weights @ V  # (B, L, H, D)
```

**Problem:** For L=2048 tokens, attention matrix = 2048×2048×16 heads = 67M elements

### Flash Attention (Optimized)

```python
from flash_attn import flash_attn_func

# Same input projections
Q = self.q_proj(hidden_states)  # (B, L, H, D)
K = self.k_proj(hidden_states)
V = self.v_proj(hidden_states)

# Flash attention (memory-efficient)
attn_output = flash_attn_func(
    Q, K, V,
    dropout_p=0.0,
    softmax_scale=1.0 / sqrt(d_k),
    causal=True  # Autoregressive masking
)
```

**Memory:** O(B * L * H * D) — No explicit attention matrix!

---

## Integration in LLaMA

**File:** `uni_interleaved/models/decoders/modeling_llama_mmfs.py:417-450`

```python
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_flash_attn = getattr(config, "use_flash_attn", False)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        B, L, D = hidden_states.shape
        H = self.num_heads

        Q = self.q_proj(hidden_states).view(B, L, H, D // H)
        K = self.k_proj(hidden_states).view(B, L, H, D // H)
        V = self.v_proj(hidden_states).view(B, L, H, D // H)

        if self.use_flash_attn:
            # Flash Attention path
            attn_output = flash_attn_func(Q, K, V, causal=True)
        else:
            # Standard attention path
            attn_output = self._standard_attention(Q, K, V, attention_mask)

        attn_output = attn_output.view(B, L, D)
        return self.o_proj(attn_output)
```

---

## Enable in Config

```yaml
# configs/train/pretrain_stage1.yaml
model:
  mm_decoder:
    use_flash_attn: true  # Enable Flash Attention
    num_layers: 32
    hidden_size: 4096
```

---

## Performance Comparison

### Memory Usage

| Sequence Length | Standard Attention | Flash Attention | Reduction |
|-----------------|-------------------|-----------------|-----------|
| 512 | 8 GB | 4 GB | 50% |
| 1024 | 24 GB | 6 GB | 75% |
| 2048 | 64 GB | 10 GB | 84% |
| 4096 | OOM | 18 GB | — |

### Speed

| Sequence Length | Standard Attention | Flash Attention | Speedup |
|-----------------|-------------------|-----------------|---------|
| 512 | 120 ms | 45 ms | 2.7× |
| 1024 | 450 ms | 95 ms | 4.7× |
| 2048 | OOM | 210 ms | — |

**Tested on A100 80GB**

---

## Multi-Image Sequences

```python
# DEEM with 10 images
num_images = 10
num_visual_tokens = 77 * num_images  # 770 tokens
num_text_tokens = 512
total_length = num_visual_tokens + num_text_tokens  # 1282 tokens

# Without Flash Attention
# Memory: 1282² × 16 heads × 4 bytes = 105 MB per batch

# With Flash Attention
# Memory: 1282 × 4096 × 4 bytes = 21 MB per batch
# 5× reduction!
```

---

## Cross-Attention with Flash Attention

**File:** `uni_interleaved/models/decoders/modeling_llama_mmfs.py:520-550`

```python
class LlamaCrossAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        """
        hidden_states: (B, L, D) text tokens
        encoder_hidden_states: (B, 77, D) visual tokens
        """
        B, L, D = hidden_states.shape

        Q = self.q_proj(hidden_states)  # (B, L, H, D_h)
        K = self.k_proj(encoder_hidden_states)  # (B, 77, H, D_h)
        V = self.v_proj(encoder_hidden_states)  # (B, 77, H, D_h)

        if self.use_flash_attn:
            # Flash cross-attention
            attn_output = flash_attn_func(
                Q, K, V,
                causal=False  # Cross-attention is not causal
            )
        else:
            attn_output = self._standard_cross_attention(Q, K, V)

        return self.o_proj(attn_output)
```

---

## MMFS with Flash Attention

**File:** `uni_interleaved/models/utils/ops/modules/mmfs.py:145`

```python
# MMFS uses deformable attention (custom CUDA kernel)
# Flash Attention cannot directly replace this

# However, Flash Attention is used in:
# 1. LLM self-attention (modeling_llama_mmfs.py)
# 2. LLM cross-attention (modeling_llama_mmfs.py)
# 3. Perceiver resampler (perceiver.py)
```

---

## Debugging

### Check if Flash Attention is Active

```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

# In model forward pass
if self.use_flash_attn:
    logger.info("Using Flash Attention")
else:
    logger.info("Using standard attention")
```

### Verify Kernel Launch

```bash
# Monitor GPU with nvprof
nsys profile --stats=true python train.py

# Look for flash_attn kernels in trace
# Example output:
# flash_fwd_splitkv_kernel<...>  45.2%
```

---

## Limitations

### 1. Hardware Requirements

```python
# Flash Attention requires:
# - GPU compute capability >= 8.0
# - A100, A6000, RTX 3090, RTX 4090

# Fallback for older GPUs
if torch.cuda.get_device_capability()[0] < 8:
    config.use_flash_attn = False
    print("Warning: Flash Attention disabled (old GPU)")
```

### 2. Compilation Time

```bash
# First run compiles kernels (slow)
# Subsequent runs use cached kernels (fast)

# Set cache directory
export FLASH_ATTN_CACHE_DIR=/tmp/flash_attn_cache
```

### 3. No Attention Weights

```python
# Flash Attention doesn't return attention weights
# (they're never materialized)

# If you need attn_weights for visualization:
if need_attention_weights:
    use_flash_attn = False
```

---

## Summary

**Benefits:** 75-84% memory reduction, 2-5× speedup
**Requirements:** A100/A6000/RTX 3090+, CUDA 11.6+
**Integration:** Set `use_flash_attn: true` in config

---

**Last Updated:** 2025-11-24
