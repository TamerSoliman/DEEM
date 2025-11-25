# Custom Dataset Integration

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

Add your own image-text data to DEEM for domain-specific fine-tuning.

---

## Step 1: Prepare Data Format

### JSON Format

```json
[
  {
    "image": "path/to/image1.jpg",
    "text": "A red car on the street",
    "task": "caption"
  },
  {
    "image": "path/to/image2.jpg",
    "question": "What color is the car?",
    "answer": "blue",
    "task": "vqa"
  }
]
```

---

## Step 2: Create Dataset Class

```python
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root, annt_file, transform):
        self.data_root = data_root
        self.transform = transform

        with open(annt_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load image
        image_path = os.path.join(self.data_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image_enc, image_dec = self.transform(image)

        return {
            "image_tensors": image_enc,
            "image_tensors_dec": image_dec,
            "text": ann["text"],
            "meta": {"task": ann["task"]}
        }
```

---

## Step 3: Create Collator

```python
class CustomCollator:
    def __init__(self, tokenizer, num_img_token=77):
        self.tokenizer = tokenizer
        self.num_img_token = num_img_token

    def __call__(self, batch):
        # Prepare images
        images = torch.stack([b["image_tensors"] for b in batch])
        images_dec = torch.stack([b["image_tensors_dec"] for b in batch])

        # Format text with image tokens
        texts = [
            f"<|soi|>{'<|image|>' * self.num_img_token}{b['text']}"
            for b in batch
        ]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="longest",
            return_tensors="pt"
        )

        return {
            "text_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "image_tensors": images,
            "image_tensors_dec": images_dec,
            "num_image_per_seq": torch.ones(len(batch), dtype=torch.long),
        }
```

---

## Step 4: Register in Config

```yaml
# custom_train.yaml
data:
  train:
    name: custom
    data_root: "./my_data/images"
    annt_file: "./my_data/annotations.json"
    collator: CustomCollator
    transform:
      aug_type: dual_numpy
      resolution: 224
      resolution2: 512
```

---

## Step 5: Add to build.py

```python
# In build_train_dataset()
elif config.name == "custom":
    dataset = CustomDataset(
        data_root=config.data_root,
        annt_file=config.annt_file,
        transform=transform
    )
```

---

## Example: Medical Images

```python
class MedicalVQADataset(Dataset):
    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load medical scan
        image = load_medical_image(ann["scan_path"])  # DICOM, etc.
        image = preprocess_medical(image)  # Normalize, resize

        # Format as VQA
        question = ann["question"]
        answer = ann["answer"]

        return {
            "image_tensors": image,
            "question": question,
            "answer": answer,
        }
```

---

## Summary

**Steps:** Format data → Create dataset → Create collator → Register → Train
**Flexibility:** Support any image-text format
**Use cases:** Domain adaptation, custom tasks

---

**Last Updated:** 2025-11-24
