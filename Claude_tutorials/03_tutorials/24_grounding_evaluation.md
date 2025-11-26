# Grounding Evaluation

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

Evaluate referring expression comprehension and grounding performance on RefCOCO, RefCOCO+, and RefCOCOg benchmarks.

---

## Datasets

### RefCOCO

**Focus:** Object categories (people, animals, objects)
**Splits:** train (120K), val (10K), testA (5K people), testB (5K objects)

```json
{
  "ref_id": 12345,
  "image_id": 123456,
  "sent": "the orange cat on the left",
  "bbox": [120, 340, 330, 440],  # [x, y, w, h]
  "category": "cat"
}
```

### RefCOCO+

**Focus:** Appearance-based (no location words like "left", "right")
**Splits:** train (120K), val (10K), testA (5K), testB (5K)

```json
{
  "sent": "the orange cat",  # No "on the left"
  "bbox": [120, 340, 330, 440]
}
```

### RefCOCOg

**Focus:** Longer, more descriptive expressions
**Splits:** train (85K), val (5K), test (5K)

```json
{
  "sent": "the orange cat with white paws sitting on a blue mat",
  "bbox": [120, 340, 330, 440]
}
```

---

## Evaluation Metrics

### 1. Accuracy @ IoU Threshold

```python
def compute_accuracy(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: List of [x1, y1, x2, y2] bboxes
    ground_truths: List of [x1, y1, x2, y2] bboxes
    iou_threshold: IoU threshold for success
    """
    correct = 0

    for pred_bbox, gt_bbox in zip(predictions, ground_truths):
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou >= iou_threshold:
            correct += 1

    accuracy = 100.0 * correct / len(predictions)
    return accuracy

# Standard thresholds
acc_50 = compute_accuracy(preds, gts, 0.5)   # Acc@0.5
acc_75 = compute_accuracy(preds, gts, 0.75)  # Acc@0.75
```

### 2. Mean IoU

```python
def compute_mean_iou(predictions, ground_truths):
    """Average IoU across all predictions"""
    ious = []

    for pred_bbox, gt_bbox in zip(predictions, ground_truths):
        iou = compute_iou(pred_bbox, gt_bbox)
        ious.append(iou)

    mean_iou = sum(ious) / len(ious)
    return mean_iou

# Usage
mean_iou = compute_mean_iou(preds, gts)
print(f"Mean IoU: {mean_iou:.3f}")
```

---

## Inference Pipeline

**File:** `evaluate.py:450-520`

```python
import torch
from PIL import Image
from tqdm import tqdm

def evaluate_grounding(model, dataset, split="val"):
    """Evaluate on RefCOCO dataset"""

    predictions = []
    ground_truths = []
    ious = []

    for sample in tqdm(dataset):
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size

        # Get referring expression
        ref_expression = sample["sent"]

        # Format prompt
        prompt = f"{ref_expression} is at <boxleft>"

        # Generate bounding box
        with torch.no_grad():
            output = model.generate_texts(
                images=image,
                prompt=prompt,
                max_new_tokens=20,
                num_beams=5,
                temperature=1.0
            )

        # Parse predicted box
        try:
            # Expected format: "0.188 0.708 0.703 1.000 <boxright>"
            bbox_str = output.split("<boxleft>")[-1].split("<boxright>")[0]
            coords = [float(x) for x in bbox_str.strip().split()]

            # Denormalize
            pred_bbox = [
                int(coords[0] * width),
                int(coords[1] * height),
                int(coords[2] * width),
                int(coords[3] * height)
            ]
        except:
            # If parsing fails, use default (empty box)
            pred_bbox = [0, 0, 0, 0]

        # Get ground truth box
        x, y, w, h = sample["bbox"]
        gt_bbox = [x, y, x + w, y + h]

        # Compute IoU
        iou = compute_iou(pred_bbox, gt_bbox)

        predictions.append(pred_bbox)
        ground_truths.append(gt_bbox)
        ious.append(iou)

    # Compute metrics
    acc_50 = compute_accuracy(predictions, ground_truths, 0.5)
    acc_75 = compute_accuracy(predictions, ground_truths, 0.75)
    mean_iou = sum(ious) / len(ious)

    print(f"{split} Results:")
    print(f"  Acc@0.5: {acc_50:.2f}%")
    print(f"  Acc@0.75: {acc_75:.2f}%")
    print(f"  Mean IoU: {mean_iou:.3f}")

    return {
        "acc_50": acc_50,
        "acc_75": acc_75,
        "mean_iou": mean_iou
    }
```

---

## Per-Category Analysis

```python
def evaluate_by_category(predictions, ground_truths, categories):
    """Evaluate per object category"""

    category_results = {}

    for category in set(categories):
        # Filter predictions for this category
        cat_preds = [p for p, c in zip(predictions, categories) if c == category]
        cat_gts = [g for g, c in zip(ground_truths, categories) if c == category]

        # Compute accuracy
        acc_50 = compute_accuracy(cat_preds, cat_gts, 0.5)
        acc_75 = compute_accuracy(cat_preds, cat_gts, 0.75)

        category_results[category] = {
            "count": len(cat_preds),
            "acc_50": acc_50,
            "acc_75": acc_75
        }

    # Print results
    print("Per-Category Results:")
    print(f"{'Category':<20} {'Count':<10} {'Acc@0.5':<10} {'Acc@0.75':<10}")
    print("-" * 50)
    for cat, results in sorted(category_results.items()):
        print(f"{cat:<20} {results['count']:<10} "
              f"{results['acc_50']:<10.2f} {results['acc_75']:<10.2f}")

    return category_results
```

---

## Error Analysis

```python
def analyze_errors(predictions, ground_truths, ious, threshold=0.5):
    """Analyze failure cases"""

    failures = []

    for i, iou in enumerate(ious):
        if iou < threshold:
            failures.append({
                "index": i,
                "iou": iou,
                "pred_bbox": predictions[i],
                "gt_bbox": ground_truths[i]
            })

    # Sort by IoU (worst first)
    failures.sort(key=lambda x: x["iou"])

    print(f"\nFound {len(failures)} failures (IoU < {threshold})")
    print("\nWorst 10 predictions:")
    print(f"{'Index':<10} {'IoU':<10} {'Pred Box':<30} {'GT Box':<30}")
    print("-" * 80)

    for fail in failures[:10]:
        print(f"{fail['index']:<10} {fail['iou']:<10.3f} "
              f"{str(fail['pred_bbox']):<30} {str(fail['gt_bbox']):<30}")

    return failures
```

---

## Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(image, pred_bbox, gt_bbox, ref_expression, iou):
    """Visualize predicted and ground truth boxes"""

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    # Draw predicted box (red)
    x1, y1, x2, y2 = pred_bbox
    pred_rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label='Prediction'
    )
    ax.add_patch(pred_rect)

    # Draw ground truth box (green)
    x1, y1, x2, y2 = gt_bbox
    gt_rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2,
        edgecolor='green',
        facecolor='none',
        linestyle='--',
        label='Ground Truth'
    )
    ax.add_patch(gt_rect)

    # Add text
    ax.text(
        10, 30,
        f"Expression: {ref_expression}\nIoU: {iou:.3f}",
        color='white',
        fontsize=12,
        backgroundcolor='black',
        alpha=0.7
    )

    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
for i in range(5):  # Show first 5 samples
    image = dataset[i]["image"]
    visualize_predictions(
        image,
        predictions[i],
        ground_truths[i],
        dataset[i]["sent"],
        ious[i]
    )
```

---

## Benchmark Comparison

### Expected Performance

| Model | RefCOCO val | RefCOCO testA | RefCOCO testB |
|-------|-------------|---------------|---------------|
| DEEM | 72.3% | 75.1% | 68.9% |
| MDETR | 86.5% | 89.5% | 81.4% |
| UniTAB | 88.1% | 91.2% | 83.8% |

**Note:** DEEM is primarily a generative model, not specialized for grounding

---

## Running Official Evaluation

```bash
# Clone RefCOCO evaluation tools
git clone https://github.com/lichengunc/refer.git

# Generate predictions
python evaluate.py \
    --dataset refcoco \
    --split val \
    --output refcoco_val_predictions.json

# Run official evaluation
cd refer
python eval_grounding.py \
    --predictions ../refcoco_val_predictions.json \
    --dataset refcoco \
    --split val
```

---

## Common Issues

### 1. Box Format Confusion

```python
# COCO format: [x, y, width, height]
coco_bbox = [120, 340, 330, 440]

# Model output format: [x1, y1, x2, y2] normalized
model_bbox = [0.188, 0.708, 0.703, 1.000]

# Convert COCO to model format
def coco_to_xyxy(bbox, width, height):
    x, y, w, h = bbox
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    return [x1, y1, x2, y2]
```

### 2. Parsing Failures

```python
# Model might generate invalid output
output = "the box is at somewhere"  # No numbers

# Robust parsing
def parse_bbox(output, default=[0, 0, 0, 0]):
    try:
        bbox_str = output.split("<boxleft>")[-1].split("<boxright>")[0]
        coords = [float(x) for x in bbox_str.strip().split()]
        if len(coords) == 4:
            return coords
    except:
        pass

    return default
```

### 3. Out-of-Bounds Predictions

```python
# Model might predict coords > 1.0 or < 0.0
def clip_bbox(bbox):
    """Clip to [0, 1] range"""
    return [max(0, min(1, x)) for x in bbox]

# Usage
pred_bbox = [0.9, -0.1, 1.2, 0.8]
pred_bbox = clip_bbox(pred_bbox)  # [0.9, 0.0, 1.0, 0.8]
```

---

## Summary

**Datasets:** RefCOCO, RefCOCO+, RefCOCOg
**Metrics:** Acc@0.5, Acc@0.75, Mean IoU
**Format:** `<boxleft> x1 y1 x2 y2 <boxright>`
**Baseline:** ~72% Acc@0.5 on RefCOCO val

---

**Last Updated:** 2025-11-24
