# Loss Function Breakdown

**Difficulty:** ⭐⭐⭐ Intermediate  
**Time:** 25 minutes

---

## Three-Headed Loss

```python
total_loss = (
    loss_txt * 1.0 +        # Text generation
    loss_img * 5.0 +        # Image generation  
    loss_sniffer * 5.0      # Diffusion feedback
)
```

---

## 1. Text Loss (loss_txt)

**Purpose:** Train text generation (VQA, captioning)

**Formula:**
```python
loss_txt = CrossEntropy(logits[:-1], targets[1:])
```

**Masking:**
- Ignore `<|image|>` tokens
- Ignore prompt (before answer)
- Ignore padding

**Example:**
```
Input:  <|soi|> <|image|>×77 Q: What? A: Cat <eos>
Target:         <|image|>×77 Q: What? A: Cat <eos> <pad>
Mask:   [-100]  [-100]×77    [-100]×5  [Cat] [<eos>] [-100]
```

---

## 2. Image Loss (loss_img)

**Purpose:** Train image generation from text

**Formula:**
```python
# Diffusion training
noise = randn_like(latent)
noisy_latent = add_noise(latent, noise, timestep)
noise_pred = unet(noisy_latent, timestep, context)
loss_img = MSE(noise_pred, noise)
```

**When active:** Only during Stage 1 (pretraining)

---

## 3. Sniffer Loss (loss_sniffer)

**Purpose:** Improve visual understanding via diffusion

**Formula:**
```python
dif_embed = proj_i(vis_embed)
sd_loss = encoder(image, dif_embed)
loss_sniffer = sd_loss * mask_region * pos_weight
```

**Effect:** Aligns visual features with SD's understanding

---

## Loss Weights Across Stages

| Stage | txt | img | sniffer |
|-------|-----|-----|---------|
| 1 (Pretrain) | 1.0 | 5.0 | 5.0 |
| 2 (VQA SFT) | 1.0 | 0.0 | 0.0 |
| 3 (Mask SFT) | 1.0 | 5.0 | 5.0 |

---

## Monitoring

```python
# Training logs should show:
print(f"loss_txt: {output['loss_txt'].item():.4f}")    # ~0.8-2.5
print(f"loss_img: {output['loss_img'].item():.4f}")    # ~0.05-0.15
print(f"loss_sniffer: {output['loss_sniffer'].item():.4f}")  # ~0.02-0.08
```

---

**Last Updated:** 2025-11-24
