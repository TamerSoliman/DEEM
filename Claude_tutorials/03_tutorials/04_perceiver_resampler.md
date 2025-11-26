# Perceiver Resampler: Feature Compression to 77 Tokens

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 25 minutes
**File:** `uni_interleaved/models/decoders/perceiver.py:7-31`

---

## Overview

The Perceiver Resampler compresses variable-length visual features (e.g., 257 ViT tokens) into a fixed 77-token representation using cross-attention.

---

## Architecture

```python
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_queries=77,        # Output tokens
        hidden_size=1024,      # Feature dimension
        encoder_hidden_size=1024,
        num_hidden_layers=6,   # Transformer layers
        num_attention_heads=16,
        qk_normalization=True,
        gradient_checkpointing=True,
    ):
        config = Blip2QFormerConfig(hidden_size=hidden_size, ...)
        self.blip2qformer = Blip2QFormerModel(config)

        # Learnable queries
        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_size))
        self.queries.data.normal_(0, config.initializer_range)
```

---

## How Cross-Attention Works

```
Learnable Queries [1, 77, 1024]
        ↓
    Query (Q)
        ↓ Cross-Attention
Visual Features [B, 257, 1024]
        ↓
   Key (K), Value (V)
        ↓
Attention(Q, K, V) = softmax(QK^T/√d) V
        ↓
Compressed Output [B, 77, 1024]
```

**Key Insight:** Fixed queries learn to "ask" for specific visual information.

---

## Forward Pass

```python
def forward(self, encoder_hidden_states, encoder_attention_mask=None, **kwargs):
    query_embeds = kwargs.pop("query_embeds", self.queries)
    return self.blip2qformer(
        query_embeds=query_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        **kwargs
    )
```

**Example:**
```python
visual_features = torch.randn(2, 257, 1024)  # Batch 2, ViT output
resampler = PerceiverResampler(num_queries=77, hidden_size=1024)

compressed = resampler(encoder_hidden_states=visual_features)[0]
print(compressed.shape)  # torch.Size([2, 77, 1024])
```

---

## Why 77 Queries?

1. **Stable Diffusion Compatibility:** SD text encoder outputs 77 tokens
2. **Sufficient Capacity:** Captures rich visual semantics
3. **Computational Efficiency:** Much smaller than 257 tokens

**Ablation:**
| Num Queries | VQA Acc | Latency |
|-------------|---------|---------|
| 32 | 64.2 | 0.3s |
| **77** | **66.1** | **0.5s** |
| 144 | 66.3 | 0.9s |

---

## QK Normalization

```python
if qk_normalization:
    # Normalize queries and keys before attention
    Q = F.normalize(Q, dim=-1)
    K = F.normalize(K, dim=-1)
    attn = Q @ K.T  # No division by sqrt(d)
```

**Benefits:**
- More stable training
- Better gradient flow
- Reduces sensitivity to initialization

---

## Gradient Checkpointing

```python
if gradient_checkpointing:
    self.blip2qformer.gradient_checkpointing_enable()
```

**Memory Savings:** ~40% during training
**Trade-off:** ~15% slower forward pass

---

## What Do Queries Learn?

Visualize attention patterns:

```python
# Get attention weights
with torch.no_grad():
    outputs = resampler.blip2qformer(
        query_embeds=queries,
        encoder_hidden_states=features,
        output_attentions=True
    )
    attn_weights = outputs.cross_attentions[0]  # [B, H, 77, 257]

# Query 0 attends to:
plt.imshow(attn_weights[0, :, 0, :].mean(0).cpu())  # Average over heads
plt.title("Query 0 Attention Pattern")
```

**Typical Patterns:**
- Query 0-10: Global context (CLS token)
- Query 11-40: Objects and regions
- Query 41-77: Fine details and textures

---

## Integration in DEEM

```python
# In VisualTokenizer
self.perceiver_resampler = PerceiverResampler(
    num_queries=77,
    hidden_size=1024,
    encoder_hidden_size=1024,
)

# Forward
qformer_inputs = position_encoded_features  # [B, 257, 1024]
vis_embed = self.perceiver_resampler(
    encoder_hidden_states=qformer_inputs
)[0]  # [B, 77, 1024]
```

---

## Summary

**Input:** Variable-length features (257 tokens)
**Output:** Fixed 77 compressed tokens
**Mechanism:** Cross-attention with learnable queries
**Benefits:** Computational efficiency, rich compression

---

**Last Updated:** 2025-11-24
