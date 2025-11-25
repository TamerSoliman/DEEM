# Dual-Resolution Transform

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time:** 20 minutes

---

## Overview

DEEM uses two image resolutions simultaneously: 224×224 for visual encoding (CLIP) and 512×512 for image generation (Stable Diffusion).

---

## Implementation

**File:** `uni_interleaved/custom_datasets/utils/build.py`

```python
class DualResolutionTransform:
    def __init__(self, resolution=224, resolution2=512, aug_type="dual_numpy"):
        self.resolution = resolution      # Encoder resolution
        self.resolution2 = resolution2    # Decoder resolution
        self.aug_type = aug_type

        # Encoder transform (224×224)
        self.transform_enc = transforms.Compose([
            transforms.Resize(resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Decoder transform (512×512)
        self.transform_dec = transforms.Compose([
            transforms.Resize(resolution2, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # SD normalization
        ])

    def __call__(self, image):
        """Transform single image to dual resolutions"""
        image_enc = self.transform_enc(image)   # (3, 224, 224)
        image_dec = self.transform_dec(image)   # (3, 512, 512)
        return image_enc, image_dec
```

---

## Usage in Datasets

```python
# Create transform
transform = DualResolutionTransform(
    resolution=224,
    resolution2=512,
    aug_type="dual_numpy"
)

# Apply to image
image = Image.open("cat.jpg").convert("RGB")
image_enc, image_dec = transform(image)

# Use in dataset
class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = load_image(idx)
        image_enc, image_dec = self.transform(image)

        return {
            "image_tensors": image_enc,      # For CLIP encoder
            "image_tensors_dec": image_dec,  # For SD decoder
            "text": self.texts[idx]
        }
```

---

## Normalization Differences

### CLIP Normalization (Encoder)

```python
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

# Per-channel RGB normalization
for c in [0, 1, 2]:
    image_enc[c] = (image_enc[c] - mean[c]) / std[c]
```

### Stable Diffusion Normalization (Decoder)

```python
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Equivalent to: pixel = (pixel - 0.5) / 0.5 = 2 * pixel - 1
# Maps [0, 1] → [-1, 1]
```

---

## Augmentation Types

### 1. No Augmentation (`aug_type="none"`)

```python
# Just resize and normalize
transform = DualResolutionTransform(aug_type="none")
```

### 2. Dual NumPy (`aug_type="dual_numpy"`)

```python
# Apply same random crop/flip to both resolutions
random_crop = RandomResizedCrop(224)
random_flip = RandomHorizontalFlip(p=0.5)

# Ensure consistency
seed = torch.initial_seed()
torch.manual_seed(seed)
image_enc = augment(image, 224)
torch.manual_seed(seed)
image_dec = augment(image, 512)
```

### 3. CLIP Augmentation (`aug_type="clip"`)

```python
# RandAugment + color jitter
transform = transforms.Compose([
    transforms.RandAugment(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

---

## Batching

```python
# In data collator
def __call__(self, batch):
    # Stack encoder images
    images_enc = torch.stack([
        sample["image_tensors"] for sample in batch
    ])  # (B, 3, 224, 224)

    # Stack decoder images
    images_dec = torch.stack([
        sample["image_tensors_dec"] for sample in batch
    ])  # (B, 3, 512, 512)

    return {
        "image_tensors": images_enc,
        "image_tensors_dec": images_dec,
        ...
    }
```

---

## Memory Considerations

```python
# Single image memory usage
encoder_memory = 3 * 224 * 224 * 4 bytes = 0.6 MB (float32)
decoder_memory = 3 * 512 * 512 * 4 bytes = 3.1 MB (float32)

# Batch of 32 images
total_memory = 32 * (0.6 + 3.1) = 118 MB
```

**Tip:** Use mixed precision (fp16) to reduce memory by 50%

---

## Common Issues

### 1. Aspect Ratio Distortion

```python
# Bad: Direct resize
image_enc = resize(image, (224, 224))  # Distorts aspect ratio

# Good: Resize then crop
image_enc = center_crop(resize(image, 224), 224)
```

### 2. Normalization Mismatch

```python
# Encoder uses CLIP normalization
# Decoder uses SD normalization
# Don't mix them!
```

### 3. Augmentation Inconsistency

```python
# Bad: Different augmentations for dual resolutions
image_enc = augment(image, 224)  # Crops top-left
image_dec = augment(image, 512)  # Crops center

# Good: Use same random seed
```

---

## Summary

**Purpose:** Encode at 224×224, generate at 512×512
**Key:** Different normalizations for CLIP vs SD
**Memory:** ~3.7 MB per image (float32)

---

**Last Updated:** 2025-11-24
