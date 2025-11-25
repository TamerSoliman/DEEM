# Data Collators Explained

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

Collators batch process samples into model inputs with proper formatting, tokenization, and tensor creation.

---

## VQACaptionTrainCollator

**File:** `uni_interleaved/custom_datasets/train/vqa_datasets.py`

### Format

```python
# Input sample
{
    "image": PIL.Image,
    "question": "What color is the cat?",
    "answer": "orange"
}

# Output sequence
"<bos> <|soi|> <|image|>×77 Question: What color is the cat? Answer: orange <eos>"
```

### Key Features
- Formats VQA as instruction-following
- Masks prompt (only supervise answer)
- Handles multi-turn dialogues

---

## ImageTextPairTrainCollator

**File:** `uni_interleaved/custom_datasets/train/pairs_datasets.py`

### Format

```python
# Captioning
"<bos> <|soi|> <|image|>×77 {caption} <eos>"

# Image-first or text-first (random)
if img_first_prob > random():
    sequence = f"<|soi|> <|image|>×77 {text}"
else:
    sequence = f"{text} <|soi|> <|image|>×77"
```

---

## GroundingTrainCollator  

**File:** `uni_interleaved/custom_datasets/train/grounding_datasets.py`

### Format

```python
# With bounding box
"<bos> <|soi|> <|image|>×77 The cat is at <boxleft> 0.3 0.4 0.7 0.8 <boxright> <eos>"
```

### Normalization

```python
# Normalize boxes to [0, 1]
x1, y1, x2, y2 = box
x1 /= image_width
x2 /= image_width  
y1 /= image_height
y2 /= image_height
```

---

## Common Methods

### Tokenization

```python
tokenized = self.tokenizer(
    texts,
    padding="longest",      # Pad to longest in batch
    max_length=2048,
    truncation=True,
    return_tensors="pt"
)
```

### Image Stacking

```python
# Dual resolution
images_enc = torch.stack([s["image_tensors"] for s in batch])      # 224×224
images_dec = torch.stack([s["image_tensors_dec"] for s in batch])  # 512×512
```

---

## Summary

**VQA:** Question-answer format with masking
**Pairs:** Flexible image-text ordering
**Grounding:** Bounding box integration

---

**Last Updated:** 2025-11-24
