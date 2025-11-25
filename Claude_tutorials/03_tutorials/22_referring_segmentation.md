# Referring Segmentation

**Difficulty:** ⭐⭐⭐⭐ Advanced
**Time:** 35 minutes

---

## Overview

Generate or predict bounding boxes for objects described in natural language using DEEM's grounding capabilities.

---

## Task Description

**Input:** Image + text description
**Output:** Bounding box coordinates

**Example:**
```
Input: "The orange cat on the left"
Output: [0.12, 0.34, 0.45, 0.78]  # [x1, y1, x2, y2] normalized
```

---

## Special Tokens

**File:** `uni_interleaved/custom_datasets/constants.py`

```python
# Grounding tokens
BOX_LEFT = "<boxleft>"   # Token ID: 32002
BOX_RIGHT = "<boxright>" # Token ID: 32003

# Format
# "<boxleft> x1 y1 x2 y2 <boxright>"
```

---

## Dataset Format

**File:** `uni_interleaved/custom_datasets/train/grounding_datasets.py`

### RefCOCO Format

```json
{
  "image": "COCO_train2014_000000123456.jpg",
  "refs": [
    {
      "sent": "the orange cat on the left",
      "bbox": [120, 340, 450, 780]  # [x, y, width, height] in pixels
    }
  ],
  "width": 640,
  "height": 480
}
```

### Normalize Boxes

```python
def normalize_bbox(bbox, image_width, image_height):
    """
    bbox: [x, y, width, height] in pixels
    Returns: [x1, y1, x2, y2] normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Convert to x1, y1, x2, y2
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # Normalize
    x1_norm = x1 / image_width
    y1_norm = y1 / image_height
    x2_norm = x2 / image_width
    y2_norm = y2 / image_height

    return [x1_norm, y1_norm, x2_norm, y2_norm]

# Example
bbox = [120, 340, 330, 440]  # [x, y, w, h]
width, height = 640, 480

bbox_norm = normalize_bbox(bbox, width, height)
# [0.1875, 0.7083, 0.7031, 1.625] → clip to [0, 1]
```

---

## Training Format

**File:** `uni_interleaved/custom_datasets/train/grounding_datasets.py:180-220`

```python
def format_grounding_sample(image, ref, width, height):
    """Format as text sequence with box"""

    # Normalize box
    bbox_norm = normalize_bbox(ref["bbox"], width, height)

    # Format sequence
    sequence = (
        f"<bos> <|soi|> <|image|>×77 "
        f"{ref['sent']} is at "
        f"<boxleft> {bbox_norm[0]:.3f} {bbox_norm[1]:.3f} "
        f"{bbox_norm[2]:.3f} {bbox_norm[3]:.3f} <boxright> <eos>"
    )

    return sequence

# Example output
# "<bos> <|soi|> <|image|>×77 the orange cat on the left is at <boxleft> 0.188 0.708 0.703 1.000 <boxright> <eos>"
```

---

## Inference

### Generate Bounding Box

```python
def predict_bbox(model, image, text_description):
    """Predict bounding box for text description"""

    # Format prompt
    prompt = f"{text_description} is at <boxleft>"

    # Generate
    with torch.no_grad():
        output = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=20,
            num_beams=3,
            temperature=1.0
        )

    # Parse output
    # Expected: "0.188 0.708 0.703 1.000 <boxright>"
    bbox_str = output.split("<boxleft>")[-1].split("<boxright>")[0]
    coords = [float(x) for x in bbox_str.strip().split()]

    # Denormalize to pixel coordinates
    width, height = image.size
    x1, y1, x2, y2 = coords
    bbox_pixels = [
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
    ]

    return bbox_pixels

# Usage
image = Image.open("cat.jpg")
bbox = predict_bbox(model, image, "the orange cat on the left")
print(f"Bounding box: {bbox}")  # [120, 340, 450, 780]
```

---

## Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bbox(image, bbox, text_description):
    """Draw bounding box on image"""

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Create rectangle
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )

    ax.add_patch(rect)
    ax.text(
        x1, y1 - 10,
        text_description,
        color='red',
        fontsize=12,
        backgroundcolor='white'
    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
image = Image.open("cat.jpg")
bbox = predict_bbox(model, image, "the orange cat")
visualize_bbox(image, bbox, "the orange cat")
```

---

## Multiple Objects

```python
def predict_multiple_bboxes(model, image, descriptions):
    """Predict boxes for multiple objects"""

    results = []
    for desc in descriptions:
        bbox = predict_bbox(model, image, desc)
        results.append({
            "description": desc,
            "bbox": bbox
        })

    return results

# Usage
image = Image.open("street_scene.jpg")
descriptions = [
    "red car on the left",
    "person walking",
    "stop sign"
]

results = predict_multiple_bboxes(model, image, descriptions)

# Visualize all boxes
fig, ax = plt.subplots(1)
ax.imshow(image)

colors = ['red', 'blue', 'green']
for i, result in enumerate(results):
    bbox = result["bbox"]
    x1, y1, x2, y2 = bbox

    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2,
        edgecolor=colors[i],
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1, y1 - 10, result["description"], color=colors[i])

plt.show()
```

---

## Evaluation Metrics

### IoU (Intersection over Union)

```python
def compute_iou(bbox1, bbox2):
    """
    bbox1, bbox2: [x1, y1, x2, y2]
    """
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute union
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union if union > 0 else 0

    return iou

# Example
pred_bbox = [120, 340, 450, 780]
gt_bbox = [115, 335, 455, 785]

iou = compute_iou(pred_bbox, gt_bbox)
print(f"IoU: {iou:.3f}")  # 0.92
```

### Accuracy @ IoU Threshold

```python
def evaluate_grounding(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: List of predicted bboxes
    ground_truths: List of ground truth bboxes
    """
    correct = 0

    for pred_bbox, gt_bbox in zip(predictions, ground_truths):
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou >= iou_threshold:
            correct += 1

    accuracy = 100 * correct / len(predictions)
    return accuracy

# Usage
accuracy_50 = evaluate_grounding(preds, gts, iou_threshold=0.5)
accuracy_75 = evaluate_grounding(preds, gts, iou_threshold=0.75)

print(f"Accuracy@0.5: {accuracy_50:.2f}%")
print(f"Accuracy@0.75: {accuracy_75:.2f}%")
```

---

## RefCOCO Benchmark

```python
def evaluate_refcoco(model, dataset, split="val"):
    """Evaluate on RefCOCO dataset"""

    predictions = []
    ground_truths = []

    for sample in tqdm(dataset):
        image = sample["image"]
        ref = sample["refs"][0]
        gt_bbox = normalize_bbox(
            ref["bbox"],
            sample["width"],
            sample["height"]
        )

        # Predict
        pred_bbox_norm = predict_bbox(model, image, ref["sent"])

        # Denormalize ground truth for comparison
        width, height = image.size
        gt_bbox_pixels = [
            int(gt_bbox[0] * width),
            int(gt_bbox[1] * height),
            int(gt_bbox[2] * width),
            int(gt_bbox[3] * height)
        ]

        predictions.append(pred_bbox_norm)
        ground_truths.append(gt_bbox_pixels)

    # Compute metrics
    acc_50 = evaluate_grounding(predictions, ground_truths, 0.5)
    acc_75 = evaluate_grounding(predictions, ground_truths, 0.75)

    print(f"RefCOCO {split} - Acc@0.5: {acc_50:.2f}%")
    print(f"RefCOCO {split} - Acc@0.75: {acc_75:.2f}%")

    return acc_50, acc_75
```

---

## Interactive Grounding

```python
# Click on image to get description
from matplotlib.patches import Rectangle

class InteractiveGrounding:
    def __init__(self, model, image):
        self.model = model
        self.image = image
        self.fig, self.ax = plt.subplots(1)
        self.ax.imshow(image)

        # Click callback
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Get description at this location
        description = self.describe_location(x, y)

        # Draw marker
        self.ax.plot(x, y, 'r*', markersize=10)
        self.ax.text(x, y - 20, description, color='red')
        self.fig.canvas.draw()

    def describe_location(self, x, y):
        """Describe what's at location (x, y)"""
        # Create small bounding box around click
        bbox = [x - 50, y - 50, x + 50, y + 50]

        # Normalize
        width, height = self.image.size
        bbox_norm = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]

        # Ask model what's in this box
        prompt = (
            f"What object is in this region: "
            f"<boxleft> {bbox_norm[0]:.3f} {bbox_norm[1]:.3f} "
            f"{bbox_norm[2]:.3f} {bbox_norm[3]:.3f} <boxright>? "
            f"Answer:"
        )

        description = self.model.generate_texts(
            images=self.image,
            prompt=prompt,
            max_new_tokens=10
        )

        return description.strip()

# Usage
image = Image.open("scene.jpg")
viewer = InteractiveGrounding(model, image)
plt.show()
```

---

## Summary

**Format:** `<boxleft> x1 y1 x2 y2 <boxright>`
**Coordinates:** Normalized [0, 1]
**Evaluation:** IoU @ 0.5, IoU @ 0.75
**Benchmark:** RefCOCO, RefCOCO+, RefCOCOg

---

**Last Updated:** 2025-11-24
