# Diffusion Feedback Mechanism

**Difficulty:** ⭐⭐⭐⭐ Advanced
**Time:** 35 minutes

---

## Overview

The "sniffer" loss uses Stable Diffusion's encoder to provide training feedback, aligning visual features with SD's rich image understanding.

---

## Conceptual Flow

```
Visual Features → Projection → SD Encoder → Loss
(from CLIP)       (proj_i)    (sniffer)    (loss_sniffer)
```

**Intuition:** If visual features contain good image information, SD should be able to use them to predict noise accurately.

---

## Implementation

**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:315-444`

### Step 1: Project Visual Embeddings

```python
# Visual tokenizer output
vis_embed = self.visual_tokenizer(images)  # (B, 77, 4096)

# Project to diffusion dimension
dif_embed = self.proj_i(vis_embed)  # (B, 77, 768)
```

**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:185`

```python
self.proj_i = nn.Linear(
    self.llm_hidden_size,  # 4096 (LLaMA)
    self.sniffer_hidden_size  # 768 (SD text encoder)
)
```

---

## Step 2: Stable Diffusion Forward Pass

**File:** `uni_interleaved/models/decoders/sd.py:257-307`

```python
def diffusion_loss(self, images_dec, dif_embed):
    """
    images_dec: (B, 3, 512, 512) ground truth images
    dif_embed: (B, 77, 768) visual features
    """

    # 1. Encode image to latent
    latent = self.vae.encode(images_dec).latent_dist.sample()
    latent = latent * 0.18215  # VAE scaling factor
    # latent: (B, 4, 64, 64)

    # 2. Sample random timestep
    timesteps = torch.randint(
        0, self.scheduler.num_train_timesteps, (B,)
    )  # e.g., timesteps=[245, 789, 123, ...]

    # 3. Add noise to latent
    noise = torch.randn_like(latent)
    noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

    # 4. Predict noise using UNet with visual context
    noise_pred = self.unet(
        noisy_latent,
        timesteps,
        encoder_hidden_states=dif_embed  # Visual features as context
    ).sample

    # 5. Compute MSE loss
    loss_sniffer = F.mse_loss(noise_pred, noise)

    return loss_sniffer
```

---

## Noise Prediction Details

### DDPM Noise Schedule

```python
# During training, sample random timestep t ∈ [0, 999]
t = torch.randint(0, 1000, (batch_size,))

# Add noise according to schedule
# α_t controls noise level (high α = less noise)
α_t = self.scheduler.alphas_cumprod[t]
noisy_latent = sqrt(α_t) * latent + sqrt(1 - α_t) * noise
```

### UNet Architecture

```python
# UNet predicts noise from noisy latent
class UNet2DConditionModel:
    def forward(self, latent, timestep, encoder_hidden_states):
        # encoder_hidden_states: (B, 77, 768) visual features

        # Downsample blocks with cross-attention
        for down_block in self.down_blocks:
            latent = down_block(latent)
            latent = cross_attention(latent, encoder_hidden_states)

        # Middle block
        latent = self.mid_block(latent, encoder_hidden_states)

        # Upsample blocks
        for up_block in self.up_blocks:
            latent = up_block(latent)
            latent = cross_attention(latent, encoder_hidden_states)

        # Output noise prediction
        noise_pred = self.conv_out(latent)  # (B, 4, 64, 64)
        return noise_pred
```

---

## Regional Masking (Optional)

**File:** `uni_interleaved/models/decoders/sd.py:289`

```python
# Only compute loss on certain regions
if mask_region is not None:
    # mask_region: (B, 1, 64, 64) binary mask
    loss_sniffer = F.mse_loss(
        noise_pred * mask_region,
        noise * mask_region,
        reduction="sum"
    ) / mask_region.sum()
```

**Use case:** Focus on foreground objects, ignore background

---

## Loss Weight Schedule

```python
# Stage 1: Pretraining
loss_sniffer_weight = 5.0

# Stage 2: VQA fine-tuning
loss_sniffer_weight = 0.0  # Disabled

# Stage 3: Mask fine-tuning
loss_sniffer_weight = 5.0  # Re-enabled
```

---

## Why It Works

### 1. Rich Visual Understanding

Stable Diffusion has been trained on billions of images and learns:
- Object shapes and textures
- Spatial relationships
- Lighting and shadows
- Semantic concepts

### 2. Cross-Attention Alignment

```python
# UNet cross-attention queries visual features
Q = latent_features  # What's in the noisy image?
K, V = visual_features  # What should be in the image?

attn = softmax(Q @ K.T / sqrt(d)) @ V
```

If visual features don't contain useful information, UNet can't predict noise well → high loss.

### 3. Gradient Flow

```python
loss_sniffer.backward()

# Gradients flow through:
# noise_pred ← UNet ← dif_embed ← proj_i ← vis_embed ← CLIP

# This improves:
# - CLIP feature quality
# - Projection layer
# - Perceiver resampler
```

---

## Monitoring

```python
# Typical loss values
print(f"loss_sniffer: {loss_sniffer.item():.4f}")

# Expected ranges:
# Step 0: ~0.15 (random features)
# Step 10000: ~0.08
# Step 50000: ~0.05
# Step 100000: ~0.02 (converged)
```

---

## Ablation Study

**From DEEM paper:**

| Training Setup | VQAv2 Accuracy |
|----------------|----------------|
| No sniffer loss | 64.2% |
| With sniffer loss | 66.8% |

**Improvement:** +2.6% accuracy from diffusion feedback

---

## Common Issues

### 1. NaN Loss

```python
# Cause: Exploding gradients in UNet
# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. High Memory Usage

```python
# Sniffer requires VAE + UNet in memory
# Memory: ~4GB for SD 1.5 UNet + VAE

# Solution: Freeze sniffer parameters
for param in model.sniffer.parameters():
    param.requires_grad = False
```

### 3. Slow Training

```python
# UNet forward pass is expensive
# Time: ~50ms per batch (vs 10ms for LLM forward)

# Solution: Reduce sniffer_weight during fine-tuning
loss_sniffer_weight = 0.0  # Disable in Stage 2
```

---

## Summary

**Purpose:** Improve visual features via SD feedback
**Method:** Predict noise from visual embeddings
**Effect:** +2.6% VQA accuracy
**Cost:** 4GB memory, 5× slower training

---

**Last Updated:** 2025-11-24
