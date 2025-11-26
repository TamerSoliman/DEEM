# Caption Metrics

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time:** 25 minutes

---

## Overview

Evaluate image captioning quality using BLEU, METEOR, ROUGE, CIDEr, and SPICE metrics.

---

## Installation

```bash
pip install pycocoevalcap==1.2
pip install nltk==3.8
```

---

## COCO Caption Evaluation

**Standard evaluation pipeline for MS-COCO Captions**

### Format

```json
{
  "annotations": [
    {
      "image_id": 123,
      "caption": "A cat sitting on a mat"
    },
    ...
  ],
  "images": [
    {"id": 123, "file_name": "COCO_val2014_000000123.jpg"},
    ...
  ]
}
```

---

## Generate Captions

```python
import torch
from PIL import Image

def generate_captions(model, image_paths):
    model.eval()
    results = []

    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert("RGB")

        # Generate caption
        with torch.no_grad():
            caption = model.generate_texts(
                images=image,
                prompt="",  # No prompt for captioning
                max_new_tokens=30,
                num_beams=5,
                temperature=1.0,
                top_p=0.9
            )

        # Clean caption
        caption = caption.strip()
        caption = caption.rstrip(".")

        # Extract image ID from filename
        image_id = int(img_path.split("_")[-1].split(".")[0])

        results.append({
            "image_id": image_id,
            "caption": caption
        })

    return results
```

---

## Compute Metrics

```python
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

# Load ground truth annotations
coco = COCO("captions_val2014.json")

# Load predictions
coco_result = coco.loadRes("captions_predictions.json")

# Create evaluator
coco_eval = COCOEvalCap(coco, coco_result)

# Evaluate
coco_eval.evaluate()

# Print metrics
print("=" * 50)
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")
print("=" * 50)
```

**Output:**
```
==================================================
Bleu_1: 0.753
Bleu_2: 0.589
Bleu_3: 0.442
Bleu_4: 0.328
METEOR: 0.273
ROUGE_L: 0.551
CIDEr: 1.042
SPICE: 0.207
==================================================
```

---

## Metric Explanations

### 1. BLEU (Bilingual Evaluation Understudy)

**Measures:** N-gram precision

```python
# BLEU-1: Unigram precision
prediction = "a cat on a mat"
reference = "a cat sitting on a mat"

# Unigrams in prediction: {a: 2, cat: 1, on: 1, mat: 1}
# Matches: {a: 2, cat: 1, on: 1, mat: 1} = 5/5 = 1.0

# BLEU-4: 4-gram precision
# prediction has 2 4-grams: ["a cat on a", "cat on a mat"]
# reference has 3 4-grams: ["a cat sitting on", "cat sitting on a", "sitting on a mat"]
# Matches: 1/2 = 0.5
```

**Range:** 0-1 (higher is better)

---

### 2. METEOR

**Measures:** Unigram precision and recall with synonyms

```python
# METEOR considers:
# 1. Exact word matches
# 2. Stem matches (e.g., "sitting" ↔ "sit")
# 3. Synonym matches (e.g., "cat" ↔ "feline")

prediction = "a feline sitting on a rug"
reference = "a cat on a mat"

# Matches: a, on, a (exact) + feline↔cat (synonym) + rug↔mat (synonym)
# Precision = 5/6 = 0.83
# Recall = 5/6 = 0.83
# F-score = 0.83
```

**Range:** 0-1 (higher is better)

---

### 3. ROUGE-L

**Measures:** Longest common subsequence

```python
# Find longest common subsequence (LCS)
prediction = "a cat on a mat"
reference = "a cat sitting on a mat"

# LCS: "a cat on a mat" (length = 5 words)
# Precision = LCS / len(prediction) = 5/5 = 1.0
# Recall = LCS / len(reference) = 5/6 = 0.83
# F-score = 2 * P * R / (P + R) = 0.91
```

**Range:** 0-1 (higher is better)

---

### 4. CIDEr (Consensus-based Image Description Evaluation)

**Measures:** TF-IDF weighted n-gram similarity

```python
# CIDEr uses TF-IDF to weight n-grams
# Common words (e.g., "a", "the") get low weight
# Rare descriptive words (e.g., "orange", "sitting") get high weight

prediction = "a cat sitting on a mat"
references = [
    "a cat on a mat",
    "orange cat on mat",
    "a cat sitting",
    "cat on a rug",
    "a feline on a mat"
]

# Compute TF-IDF for each n-gram
# CIDEr = average cosine similarity across all references
# CIDEr = 1.2 (typical range: 0-10)
```

**Range:** 0-10 (higher is better)
**Note:** CIDEr is the most important metric for captioning

---

### 5. SPICE (Semantic Propositional Image Caption Evaluation)

**Measures:** Semantic graph similarity

```python
# SPICE parses captions into scene graphs
prediction = "a cat sitting on a mat"
# Graph: cat(object), mat(object), sitting(relation), on(relation)

reference = "a cat on a mat"
# Graph: cat(object), mat(object), on(relation)

# Match objects: cat ✓, mat ✓
# Match relations: on ✓, sitting ✗
# SPICE = (2+1) / (2+2) = 0.75
```

**Range:** 0-1 (higher is better)

---

## Per-Image Analysis

```python
# Get per-image scores
image_scores = {}
for image_id in coco.getImgIds():
    # Get predictions for this image
    pred = coco_result.imgToAnns[image_id][0]["caption"]

    # Get references
    refs = [ann["caption"] for ann in coco.imgToAnns[image_id]]

    # Compute metrics (simplified example)
    bleu = compute_bleu(pred, refs)
    cider = compute_cider(pred, refs)

    image_scores[image_id] = {
        "bleu": bleu,
        "cider": cider
    }

# Find best/worst captions
sorted_images = sorted(
    image_scores.items(),
    key=lambda x: x[1]["cider"],
    reverse=True
)

print("Best caption:")
print(f"Image: {sorted_images[0][0]}, CIDEr: {sorted_images[0][1]['cider']:.3f}")

print("Worst caption:")
print(f"Image: {sorted_images[-1][0]}, CIDEr: {sorted_images[-1][1]['cider']:.3f}")
```

---

## Sampling Strategies

### 1. Greedy Decoding (Baseline)

```python
caption = model.generate_texts(
    images=image,
    max_new_tokens=30,
    do_sample=False  # Greedy
)
```

**Pros:** Fast, deterministic
**Cons:** Repetitive, generic captions

---

### 2. Beam Search (Best)

```python
caption = model.generate_texts(
    images=image,
    max_new_tokens=30,
    num_beams=5,  # Search top 5 candidates
    do_sample=False
)
```

**Pros:** Higher BLEU/CIDEr scores
**Cons:** Slower, still somewhat generic

---

### 3. Nucleus Sampling

```python
caption = model.generate_texts(
    images=image,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.9,  # Nucleus sampling
    temperature=0.7
)
```

**Pros:** More diverse, creative captions
**Cons:** Lower BLEU/CIDEr, less safe

---

## Expected Performance

### DEEM Baseline

| Metric | DEEM | LLaVA-1.5 | BLIP-2 |
|--------|------|-----------|--------|
| BLEU-4 | 32.8 | 35.2 | 38.1 |
| METEOR | 27.3 | 29.1 | 31.5 |
| CIDEr | 104.2 | 112.5 | 118.3 |
| SPICE | 20.7 | 22.4 | 24.1 |

---

## Debugging Low Scores

### 1. Check Caption Quality

```python
# Generate samples and inspect
for i in range(10):
    image_id = dataset[i]["image_id"]
    prediction = predictions[i]["caption"]
    references = dataset[i]["captions"]

    print(f"Image: {image_id}")
    print(f"Pred: {prediction}")
    print(f"Refs: {references}")
    print()
```

**Common issues:**
- Too short ("a cat")
- Too long ("a cat sitting on a mat in a room with a window...")
- Generic ("an image of something")
- Repetitive ("a cat cat cat")

### 2. Tune Decoding Parameters

```python
# Try different configurations
configs = [
    {"num_beams": 3, "max_new_tokens": 20},
    {"num_beams": 5, "max_new_tokens": 30},
    {"num_beams": 10, "max_new_tokens": 40},
]

for config in configs:
    captions = generate_captions(model, images, **config)
    scores = evaluate(captions)
    print(f"Config: {config}, CIDEr: {scores['CIDEr']:.3f}")
```

---

## Summary

**Metrics:** BLEU, METEOR, ROUGE, CIDEr, SPICE
**Best strategy:** Beam search (num_beams=5)
**Key metric:** CIDEr (0-10, higher is better)

---

**Last Updated:** 2025-11-24
