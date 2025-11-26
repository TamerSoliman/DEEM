# Test-Time Adaptation (TTA)

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 25 minutes

---

## Overview

Test-time adaptation fine-tunes the model on a few examples from the target domain during inference, improving accuracy on domain-specific tasks.

---

## When to Use TTA

**Use cases:**
- Medical imaging (train on natural images, adapt to X-rays)
- Domain shift (train on COCO, adapt to sketch images)
- Few-shot learning (5-10 examples of new task)

**Don't use when:**
- Already trained on target domain
- No labeled examples available
- Inference speed critical

---

## Implementation

**File:** `uni_interleaved/models/uni_interleaved.py:871`

### Step 1: Prepare Adaptation Data

```python
# Few-shot examples from target domain
adaptation_data = [
    {
        "image": medical_scan1,
        "question": "Is there a fracture?",
        "answer": "yes"
    },
    {
        "image": medical_scan2,
        "question": "Where is the tumor?",
        "answer": "left lung"
    },
    # ... 5-10 examples
]
```

---

## Step 2: Create Adapter

```python
from peft import LoraConfig, get_peft_model

# Use LoRA for efficient adaptation
lora_config = LoraConfig(
    r=8,                    # Small rank for fast adaptation
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,       # No dropout for few examples
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply to model
model.mm_decoder = get_peft_model(model.mm_decoder, lora_config)
```

---

## Step 3: Adapt on Examples

```python
import torch.optim as optim

# Freeze all except LoRA
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# Optimizer for adaptation
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.01
)

# Adapt for a few steps
model.train()
for epoch in range(5):  # 5 epochs over few examples
    for batch in adaptation_dataloader:
        outputs = model(
            text_ids=batch["text_ids"],
            image_tensors=batch["image_tensors"],
            image_tensors_dec=batch["image_tensors_dec"],
            num_image_per_seq=batch["num_image_per_seq"]
        )

        loss = outputs["loss_txt"]  # Only text loss for VQA

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Adaptation loss: {loss.item():.4f}")
```

---

## Step 4: Inference

```python
# Switch to eval mode
model.eval()

# Run inference on target domain
with torch.no_grad():
    answer = model.generate_texts(
        images=target_image,
        prompt="Question: Is there a fracture? Answer:",
        max_new_tokens=20
    )

print(f"Answer: {answer}")
```

---

## Advanced: Entropy Minimization

**Idea:** Adapt model to be more confident on unlabeled test data

```python
def entropy_loss(logits):
    """Minimize prediction entropy"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()

# Adaptation loop
for batch in unlabeled_dataloader:
    outputs = model(...)

    # Entropy minimization loss
    ent_loss = entropy_loss(outputs["logits"])

    # Also use few labeled examples
    if batch_has_labels:
        ce_loss = outputs["loss_txt"]
        loss = ce_loss + 0.1 * ent_loss
    else:
        loss = ent_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Memory-Efficient TTA

### Option 1: Adapt Only Cross-Attention

```python
# Freeze everything except cross-attention
for name, param in model.named_parameters():
    if "llama_cross_attn" not in name:
        param.requires_grad = False

# Much faster adaptation (only 77M params)
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

### Option 2: Adapt Visual Tokenizer Only

```python
# Only adapt visual encoding
for name, param in model.named_parameters():
    if "visual_tokenizer" not in name:
        param.requires_grad = False

# Useful for domain shift in images (medical → natural)
```

---

## Continual Adaptation

**Scenario:** Adapt continuously as new examples arrive

```python
# Start with pretrained model
model.load_state_dict(torch.load("pretrained.pth"))

# Adapt on each mini-batch during inference
model.train()
for i, batch in enumerate(test_dataloader):
    # Adapt if we have labels
    if batch["answer"] is not None:
        outputs = model(...)
        loss = outputs["loss_txt"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        answer = model.generate_texts(...)
    model.train()

    # Reset model periodically to avoid catastrophic forgetting
    if i % 100 == 0:
        model.load_state_dict(torch.load("pretrained.pth"))
```

---

## Evaluation

```python
# Compare baseline vs adapted
baseline_accuracy = evaluate(model_pretrained, test_data)
adapted_accuracy = evaluate(model_adapted, test_data)

print(f"Baseline: {baseline_accuracy:.2f}%")
print(f"Adapted: {adapted_accuracy:.2f}%")
print(f"Improvement: {adapted_accuracy - baseline_accuracy:.2f}%")
```

**Expected improvements:**
- Medical imaging: +5-10% accuracy with 10 examples
- Domain shift: +3-7% accuracy
- Few-shot learning: +10-20% accuracy

---

## Common Issues

### 1. Overfitting on Few Examples

```python
# Solution: Strong regularization
optimizer = optim.AdamW(params, lr=1e-4, weight_decay=0.1)  # High weight decay

# Early stopping
if val_loss > best_val_loss:
    break
```

### 2. Catastrophic Forgetting

```python
# Solution: Elastic Weight Consolidation (EWC)
# Penalize changes to important parameters

ewc_loss = sum(
    fisher_info[name] * (param - pretrained_param[name]) ** 2
    for name, param in model.named_parameters()
)

loss = task_loss + 0.1 * ewc_loss
```

### 3. Slow Adaptation

```python
# Solution: Use smaller LoRA rank
lora_config = LoraConfig(r=4)  # Instead of r=16

# Adapt for fewer steps
for epoch in range(3):  # Instead of 10
    ...
```

---

## Summary

**Purpose:** Adapt to new domain with few examples
**Method:** LoRA + few-shot fine-tuning
**Speed:** 5 epochs over 10 examples = ~30 seconds
**Improvement:** +5-10% accuracy on target domain

---

**Last Updated:** 2025-11-24
