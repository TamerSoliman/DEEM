# Image Editing

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

Use DEEM for image editing tasks: inpainting, style transfer, object removal, and attribute editing.

---

## Basic Editing Pipeline

**File:** `inference.py:250-320`

```python
from PIL import Image
import torch

def edit_image(model, source_image, editing_prompt):
    """
    Edit an image based on text prompt

    source_image: PIL.Image
    editing_prompt: str describing the edit
    """

    # Generate edited image
    edited_image = model.generate_images(
        images=source_image,
        prompt=editing_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=0.8  # How much to change (0.0=no change, 1.0=complete change)
    )

    return edited_image

# Usage
source = Image.open("cat_on_mat.jpg")
edited = edit_image(model, source, "a dog on a mat")
edited.save("dog_on_mat.jpg")
```

---

## Editing Strength

```python
# Lower strength = preserve more of original
strengths = [0.3, 0.5, 0.7, 0.9]

for strength in strengths:
    edited = model.generate_images(
        images=source_image,
        prompt="a watercolor painting",
        strength=strength,
        num_inference_steps=50
    )
    edited.save(f"edited_strength_{strength}.jpg")

# strength=0.3: Subtle changes
# strength=0.5: Moderate changes
# strength=0.7: Significant changes
# strength=0.9: Nearly new image
```

---

## Common Editing Tasks

### 1. Object Replacement

```python
# Replace cat with dog
source = Image.open("cat_on_mat.jpg")

edited = model.generate_images(
    images=source,
    prompt="a golden retriever dog on a mat, same background",
    strength=0.7,
    guidance_scale=7.5
)

edited.save("dog_on_mat.jpg")
```

### 2. Style Transfer

```python
# Convert to different artistic style
source = Image.open("photo.jpg")

styles = [
    "oil painting style",
    "watercolor painting",
    "pencil sketch",
    "anime style",
    "3D render"
]

for style in styles:
    edited = model.generate_images(
        images=source,
        prompt=f"same content as input, {style}",
        strength=0.6,
        guidance_scale=7.5
    )
    edited.save(f"style_{style.replace(' ', '_')}.jpg")
```

### 3. Attribute Editing

```python
# Change specific attributes
source = Image.open("person.jpg")

edits = [
    "same person, smiling",
    "same person, with sunglasses",
    "same person, blonde hair",
    "same person, wearing a hat"
]

for edit in edits:
    edited = model.generate_images(
        images=source,
        prompt=edit,
        strength=0.5,  # Lower strength to preserve identity
        guidance_scale=7.5
    )
    edited.save(f"edit_{edit[:20]}.jpg")
```

### 4. Background Change

```python
# Keep subject, change background
source = Image.open("person_indoors.jpg")

edited = model.generate_images(
    images=source,
    prompt="same person, standing on a beach at sunset",
    strength=0.6,
    guidance_scale=7.5
)

edited.save("person_on_beach.jpg")
```

---

## Inpainting with Masks

**Note:** DEEM doesn't have explicit inpainting support, but can approximate it

```python
def inpaint_region(model, image, mask, fill_prompt):
    """
    Inpaint masked region

    image: PIL.Image
    mask: PIL.Image (white = inpaint, black = keep)
    fill_prompt: str describing what to fill
    """

    # Composite approach: Use strong editing with prompt
    edited = model.generate_images(
        images=image,
        prompt=fill_prompt,
        strength=0.9,  # High strength for masked region
        guidance_scale=7.5
    )

    # Blend using mask (manual post-processing)
    import numpy as np

    img_np = np.array(image).astype(float)
    edited_np = np.array(edited).astype(float)
    mask_np = np.array(mask.convert("L")).astype(float) / 255.0
    mask_np = mask_np[:, :, None]  # Add channel dimension

    # Blend: masked regions from edited, rest from original
    result_np = img_np * (1 - mask_np) + edited_np * mask_np
    result = Image.fromarray(result_np.astype(np.uint8))

    return result
```

---

## Multi-Step Editing

```python
# Sequential edits
image = Image.open("scene.jpg")

# Step 1: Change time of day
image = edit_image(image, "same scene at sunset", strength=0.6)

# Step 2: Add object
image = edit_image(image, "add a person walking", strength=0.5)

# Step 3: Stylize
image = edit_image(image, "cinematic style", strength=0.4)

image.save("final_edited.jpg")
```

---

## Guidance Scale

```python
# Control adherence to prompt
guidance_scales = [1.0, 3.0, 7.5, 15.0]

for scale in guidance_scales:
    edited = model.generate_images(
        images=source,
        prompt="a painting",
        guidance_scale=scale,
        strength=0.7
    )
    edited.save(f"guidance_{scale}.jpg")

# scale=1.0: Ignore prompt, mostly original
# scale=3.0: Light guidance
# scale=7.5: Balanced (default)
# scale=15.0: Strong guidance, may be over-saturated
```

---

## Deterministic Editing

```python
# Use same seed for reproducible results
import torch

def deterministic_edit(model, image, prompt, seed=42):
    """Reproducible image editing"""

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    edited = model.generate_images(
        images=image,
        prompt=prompt,
        strength=0.7,
        guidance_scale=7.5
    )

    return edited

# Always produces same result
edited1 = deterministic_edit(model, image, "a dog", seed=42)
edited2 = deterministic_edit(model, image, "a dog", seed=42)
# edited1 == edited2
```

---

## Iterative Refinement

```python
def iterative_edit(model, image, prompt, iterations=3):
    """Gradually refine edit over multiple steps"""

    current_image = image
    strength = 0.4  # Start with low strength

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")

        current_image = model.generate_images(
            images=current_image,
            prompt=prompt,
            strength=strength,
            guidance_scale=7.5
        )

        # Increase strength gradually
        strength = min(0.9, strength + 0.1)

    return current_image

# Usage
source = Image.open("photo.jpg")
edited = iterative_edit(model, source, "oil painting", iterations=3)
edited.save("refined_painting.jpg")
```

---

## Evaluation

### Visual Similarity

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def evaluate_edit(original, edited):
    """Measure how much image changed"""

    # Convert to numpy
    orig_np = np.array(original.convert("RGB"))
    edit_np = np.array(edited.convert("RGB"))

    # Compute SSIM
    similarity = ssim(orig_np, edit_np, channel_axis=2)

    print(f"SSIM: {similarity:.3f}")
    # High SSIM (>0.8) = small changes
    # Low SSIM (<0.5) = large changes

    return similarity
```

### Text Alignment

```python
import clip

def evaluate_text_alignment(edited_image, target_prompt):
    """Measure how well edit matches prompt"""

    model, preprocess = clip.load("ViT-B/32", device="cuda")

    image_input = preprocess(edited_image).unsqueeze(0).cuda()
    text_input = clip.tokenize([target_prompt]).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    print(f"CLIP Score: {similarity:.3f}")
    # High score (>0.3) = good alignment
    # Low score (<0.2) = poor alignment

    return similarity
```

---

## Batch Editing

```python
from torch.utils.data import DataLoader

def batch_edit(model, images, prompts, batch_size=4):
    """Edit multiple images in parallel"""

    results = []

    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        # Generate batch
        with torch.no_grad():
            edited_batch = model.generate_images_batch(
                images=batch_images,
                prompts=batch_prompts,
                strength=0.7,
                guidance_scale=7.5
            )

        results.extend(edited_batch)

    return results

# Usage
images = [Image.open(f"img_{i}.jpg") for i in range(100)]
prompts = [f"make it artistic style {i}" for i in range(100)]

edited_images = batch_edit(model, images, prompts)
```

---

## Common Issues

### 1. Over-Editing

```python
# Problem: Edit changes too much
# Solution: Lower strength

edited = model.generate_images(
    images=source,
    prompt="make it blue",
    strength=0.3,  # Lower strength
    guidance_scale=5.0  # Lower guidance
)
```

### 2. Under-Editing

```python
# Problem: Edit doesn't change enough
# Solution: Higher strength

edited = model.generate_images(
    images=source,
    prompt="completely different scene",
    strength=0.9,  # Higher strength
    guidance_scale=10.0  # Higher guidance
)
```

### 3. Artifacts

```python
# Problem: Generated image has artifacts
# Solution: More inference steps

edited = model.generate_images(
    images=source,
    prompt="edit",
    num_inference_steps=100,  # More steps (default 50)
    strength=0.7
)
```

---

## Summary

**Basic editing:** strength=0.7, guidance_scale=7.5
**Subtle edits:** strength=0.3-0.5
**Major edits:** strength=0.8-0.9
**Best quality:** num_inference_steps=100

---

**Last Updated:** 2025-11-24
