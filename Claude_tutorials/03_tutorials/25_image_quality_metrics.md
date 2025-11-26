# Image Quality Metrics

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time:** 25 minutes

---

## Overview

Evaluate generated image quality using FID, IS, CLIP Score, and perceptual metrics.

---

## Installation

```bash
pip install torch-fidelity==0.3.0
pip install lpips==0.1.4
pip install clean-fid==0.1.35
```

---

## 1. FID (Fréchet Inception Distance)

**Measures:** Distribution similarity between real and generated images

### Compute FID

```python
from pytorch_fid import fid_score

# Compute FID between two image directories
fid_value = fid_score.calculate_fid_given_paths(
    paths=["./real_images", "./generated_images"],
    batch_size=50,
    device="cuda",
    dims=2048  # Inception feature dimension
)

print(f"FID: {fid_value:.2f}")
```

**Interpretation:**
- FID = 0: Perfect match
- FID < 10: Excellent quality
- FID < 50: Good quality
- FID > 100: Poor quality

---

## 2. Inception Score (IS)

**Measures:** Diversity and quality of generated images

```python
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="./generated_images",
    cuda=True,
    isc=True,  # Compute Inception Score
    batch_size=50
)

print(f"Inception Score: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")
```

**Interpretation:**
- IS > 10: Excellent diversity and quality
- IS > 5: Good
- IS < 3: Poor

---

## 3. CLIP Score

**Measures:** Text-image alignment

```python
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_score(image, text):
    """Compute CLIP score for single image-text pair"""

    # Preprocess
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    clip_score = (image_features @ text_features.T).item()

    return clip_score

# Usage
image = Image.open("generated_cat.jpg")
text = "a cat sitting on a mat"
score = compute_clip_score(image, text)
print(f"CLIP Score: {score:.3f}")  # Range: [-1, 1], higher is better
```

### Batch CLIP Score

```python
def compute_clip_scores_batch(images, texts):
    """Compute CLIP scores for multiple image-text pairs"""

    scores = []
    for image, text in zip(images, texts):
        score = compute_clip_score(image, text)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score, scores

# Usage
images = [Image.open(f"gen_{i}.jpg") for i in range(100)]
texts = [f"a photo of {i}" for i in range(100)]

mean_clip_score, individual_scores = compute_clip_scores_batch(images, texts)
print(f"Mean CLIP Score: {mean_clip_score:.3f}")
```

---

## 4. LPIPS (Learned Perceptual Image Patch Similarity)

**Measures:** Perceptual similarity between two images

```python
import lpips

# Load LPIPS model
loss_fn = lpips.LPIPS(net='alex').cuda()  # or 'vgg'

def compute_lpips(image1, image2):
    """
    Compute perceptual distance between two images
    Lower is better (more similar)
    """

    # Convert to tensor [-1, 1]
    img1_tensor = lpips.im2tensor(image1).cuda()
    img2_tensor = lpips.im2tensor(image2).cuda()

    # Compute distance
    distance = loss_fn(img1_tensor, img2_tensor)

    return distance.item()

# Usage
original = Image.open("original.jpg")
edited = Image.open("edited.jpg")

lpips_dist = compute_lpips(original, edited)
print(f"LPIPS Distance: {lpips_dist:.3f}")  # Range: [0, 1], lower is better
```

---

## 5. SSIM (Structural Similarity Index)

**Measures:** Structural similarity between images

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(image1, image2):
    """
    Compute SSIM between two images
    Higher is better (more similar)
    """

    # Convert to grayscale
    img1_gray = np.array(image1.convert("L"))
    img2_gray = np.array(image2.convert("L"))

    # Compute SSIM
    similarity = ssim(img1_gray, img2_gray)

    return similarity

# Usage
original = Image.open("original.jpg")
edited = Image.open("edited.jpg")

ssim_score = compute_ssim(original, edited)
print(f"SSIM: {ssim_score:.3f}")  # Range: [-1, 1], higher is better
```

---

## 6. PSNR (Peak Signal-to-Noise Ratio)

**Measures:** Pixel-level difference

```python
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(image1, image2):
    """
    Compute PSNR between two images
    Higher is better (more similar)
    """

    # Convert to numpy arrays
    img1 = np.array(image1)
    img2 = np.array(image2)

    # Compute PSNR
    psnr_value = psnr(img1, img2, data_range=255)

    return psnr_value

# Usage
original = Image.open("original.jpg")
compressed = Image.open("compressed.jpg")

psnr_score = compute_psnr(original, compressed)
print(f"PSNR: {psnr_score:.2f} dB")  # Higher is better
```

---

## Comprehensive Evaluation

```python
def evaluate_generated_images(
    generated_images_dir,
    real_images_dir,
    prompts=None
):
    """Comprehensive image quality evaluation"""

    # 1. FID Score
    print("Computing FID...")
    fid = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size=50,
        device="cuda",
        dims=2048
    )

    # 2. Inception Score
    print("Computing IS...")
    metrics = calculate_metrics(
        input1=generated_images_dir,
        cuda=True,
        isc=True
    )
    inception_score = metrics["inception_score_mean"]

    # 3. CLIP Score (if prompts available)
    clip_score = None
    if prompts:
        print("Computing CLIP Score...")
        images = [Image.open(f"{generated_images_dir}/{i}.jpg") for i in range(len(prompts))]
        clip_score, _ = compute_clip_scores_batch(images, prompts)

    # Print results
    print("\n" + "=" * 50)
    print("Image Quality Metrics")
    print("=" * 50)
    print(f"FID: {fid:.2f} (lower is better)")
    print(f"Inception Score: {inception_score:.2f} (higher is better)")
    if clip_score:
        print(f"CLIP Score: {clip_score:.3f} (higher is better)")
    print("=" * 50)

    return {
        "fid": fid,
        "inception_score": inception_score,
        "clip_score": clip_score
    }
```

---

## Evaluate DEEM Generations

```python
# Generate images with DEEM
def generate_images_for_evaluation(model, prompts, output_dir):
    """Generate images for evaluation"""

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(tqdm(prompts)):
        # Generate image
        image = model.generate_images(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        )

        # Save
        image.save(f"{output_dir}/{i:05d}.jpg")

# Load prompts
with open("coco_captions.txt") as f:
    prompts = [line.strip() for line in f.readlines()][:1000]

# Generate
generate_images_for_evaluation(model, prompts, "./generated")

# Evaluate
results = evaluate_generated_images(
    generated_images_dir="./generated",
    real_images_dir="./coco_val2014",
    prompts=prompts
)
```

---

## Benchmark Comparison

### Expected Performance

| Model | FID ↓ | IS ↑ | CLIP Score ↑ |
|-------|-------|------|--------------|
| DEEM | 18.5 | 32.1 | 0.28 |
| Stable Diffusion 1.5 | 12.3 | 38.5 | 0.31 |
| DALL-E 2 | 10.2 | 42.1 | 0.34 |
| Imagen | 7.3 | 45.8 | 0.36 |

---

## Per-Prompt Analysis

```python
# Analyze which prompts generate best images
def analyze_per_prompt(images, prompts):
    """Find best and worst prompts by CLIP score"""

    scores = []
    for image, prompt in zip(images, prompts):
        score = compute_clip_score(image, prompt)
        scores.append((prompt, score))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 prompts:")
    for prompt, score in scores[:10]:
        print(f"{score:.3f}: {prompt}")

    print("\nWorst 10 prompts:")
    for prompt, score in scores[-10:]:
        print(f"{score:.3f}: {prompt}")

    return scores
```

---

## Common Issues

### 1. FID Requires Many Images

```python
# FID needs at least 2048 images for reliable results
# If you have fewer images, use Clean-FID instead

from cleanfid import fid

fid_value = fid.compute_fid(
    "./generated",
    "./real",
    mode="clean",
    num_workers=4,
    batch_size=32
)
```

### 2. Image Size Mismatch

```python
# Resize all images to same size before computing metrics
from torchvision import transforms

resize_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512)
])

for img_path in image_paths:
    img = Image.open(img_path)
    img = resize_transform(img)
    img.save(img_path)
```

---

## Summary

**FID:** Distribution similarity (lower is better)
**IS:** Diversity and quality (higher is better)
**CLIP Score:** Text-image alignment (higher is better)
**LPIPS:** Perceptual similarity (lower is better)

---

**Last Updated:** 2025-11-24
