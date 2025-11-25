# Image Generation Pipeline in DEEM

**Difficulty:** ⭐⭐⭐⭐ Advanced
**Time:** 35 minutes
**File:** `uni_interleaved/models/uni_interleaved.py:592-681`

---

## Overview

DEEM generates images using Stable Diffusion 2.1 conditioned on LLM context and multi-scale visual features via MMFS injection.

---

## Architecture

```
Text Context
    ↓
[LLM Forward] → Context Features [N_img, L_ctx, 4096]
    ↓
[Perceiver Resampler] → Compressed [N_img, 77, 1024]
    ↓                                          ↓
[Stable Diffusion VAE Decoder]    [Multi-scale Features]
    ↓                                          ↓
Latent Space [N_img, 4, 64, 64]    [MMFS Injection to UNet]
    ↓
[Iterative Denoising 50 steps]
    ↓
Final Latent
    ↓
[VAE Decode]
    ↓
Generated Image [N_img, 3, 512, 512]
```

---

## Step-by-Step Process

### Step 1: Extract Context Features

**Code:** `uni_interleaved.py:550-558`

```python
(context_features, context_attention_mask) = \
    self._prepare_context_features_for_image_decoder(
        mm_hidden_state,  # LLM outputs
        text_ids=text_ids,
        nearest_bos_idxs=nearest_bos_idxs,
    )
# Output: [N_images, L_context, 4096]
```

**What it does:** Extracts text before each `<|soi|>` token

**Example:**
```
Sequence: "A red cat <|soi|> <|image|>×77"
Context: ["<|soi|>", "cat", "red", "A"]  (reversed)
```

---

### Step 2: Prepare MMFS Features

```python
(mmfs_features, mmfs_mask) = \
    self._prepare_mmfs_features_for_image_decoder(
        multiscale_features,
        text_ids=text_ids,
        nearest_bos_idxs=nearest_bos_idxs,
        num_image_per_seq=num_image_per_seq,
    )
# mmfs_features: [N_img, 1, ...] from previous image
```

**Purpose:** For interleaved generation, condition on previous images

---

### Step 3: Image Decoder Forward

**File:** `uni_interleaved/models/decoders/decoder_image.py`

```python
output = self.image_decoder.generate_images(
    decoder=self.visual_tokenizer.encoder,  # SD instance
    context_features=context_features,
    context_attention_mask=context_attention_mask,
    mmfs_features=mmfs_features,
    mmfs_mask=mmfs_mask,
    num_inference_steps=50,
    guidance_scale=7.5,
)
# Returns: {"image": [N_img, 3, 512, 512]}
```

---

## Inside Stable Diffusion Generation

### Step 1: Context Compression

```python
# Compress context with Perceiver
encoder_hidden_states = self.perceiver_resampler(
    encoder_hidden_states=context_features,
    encoder_attention_mask=context_attention_mask,
)[0]  # [N_img, 77, 1024]
```

**Replaces:** SD's CLIP text encoder output (77 tokens)

---

### Step 2: Classifier-Free Guidance Setup

```python
# Unconditional embeddings (for CFG)
negative_prompt_embeds = self.neg_prompt_embeds.expand_as(encoder_hidden_states)

# Concatenate for CFG
encoder_hidden_states = torch.cat([
    negative_prompt_embeds,  # Unconditional
    encoder_hidden_states    # Conditional
])  # [2*N_img, 77, 1024]
```

---

### Step 3: Initialize Latents

```python
# Random noise in latent space
latents = torch.randn(
    N_img, 4, 64, 64,  # 4 channels, 64×64 latent
    generator=generator,
    device=device
) * scheduler.init_noise_sigma
```

---

### Step 4: Denoising Loop

```python
for t in tqdm(scheduler.timesteps):
    # Expand for CFG
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # UNet forward with MMFS injection
    noise_pred = decoder.unet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_hidden_states,
        mmfs_features=mmfs_features,  # Injected here!
        mmfs_mask=mmfs_mask,
    ).sample

    # CFG: combine conditional and unconditional
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # Denoise step
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

---

### Step 5: Decode to Pixel Space

```python
# Scale latents
latents = 1 / 0.18215 * latents

# VAE decode
image = decoder.vae.decode(latents).sample

# Clamp to [0, 1]
image = (image / 2 + 0.5).clamp(0, 1)
```

---

## MMFS Injection Details

**File:** `uni_interleaved/models/decoders/sd_mmfs.py`

```python
class MMFSNet(nn.Module):
    def forward(self, unet_features, mmfs_features, mmfs_mask):
        # Inject at matching spatial resolutions
        for scale_idx, unet_feat in enumerate(unet_features):
            if unet_feat.shape[-1] in [64, 32, 16, 8]:
                # Fuse MMFS features
                mmfs_inject = self.mmfs_blocks[scale_idx](
                    unet_feat,
                    mmfs_features[scale_idx],
                    mmfs_mask
                )
                unet_feat = unet_feat + mmfs_inject
        return unet_features
```

**Purpose:** Inject multi-scale visual features for consistency

---

## Example: Text-to-Image

```python
prompt = "A majestic lion in the savanna at sunset"
text = f"{prompt}<|soi|>{'<|image|>' * 77}"

tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate_images(
        text_ids=tokens.input_ids.to("cuda"),
        image_tensors=torch.zeros(1, 3, 224, 224).to("cuda"),  # Dummy
        image_tensors_dec=torch.zeros(1, 3, 512, 512).to("cuda"),
        num_image_per_seq=torch.tensor([1]),
        attention_mask=tokens.attention_mask.to("cuda"),
        target_image_idxs=torch.tensor([0]),
        num_inference_steps=50,
        guidance_scale=7.5,
    )

from torchvision.utils import save_image
save_image(outputs["image"][0], "lion_sunset.png")
```

---

## Hyperparameters

### `num_inference_steps`

- **25 steps:** Fast, lower quality
- **50 steps:** Balanced (default)
- **100 steps:** High quality, slow

### `guidance_scale`

- **1.0:** No CFG (fast but ignores prompt)
- **7.5:** Balanced (default)
- **15.0:** Strong adherence to prompt, may oversaturate

---

## Performance

**Generation Time (A100):**
- 25 steps: ~500ms
- 50 steps: ~1s
- 100 steps: ~2s

**Batch Processing:**
- 1 image: 1s
- 4 images: 1.5s (batched)
- 8 images: 2s

---

## Common Issues

### Issue 1: Poor Image Quality

**Solutions:**
- Increase `num_inference_steps` to 100
- Increase `guidance_scale` to 10-15
- Check context features aren't collapsed

### Issue 2: Ignores Text Prompt

**Cause:** CFG scale too low

**Solution:**
```python
guidance_scale=10.0  # Increase from 7.5
```

### Issue 3: Artifacts

**Cause:** Extreme guidance or too few steps

**Solution:**
```python
num_inference_steps=75
guidance_scale=7.5
```

---

## Summary

**Pipeline:** Context → Perceiver → SD Latent → MMFS UNet → Decode
**Key Parameters:** Steps (50), CFG scale (7.5)
**Output:** 512×512 RGB images

---

**Last Updated:** 2025-11-24
