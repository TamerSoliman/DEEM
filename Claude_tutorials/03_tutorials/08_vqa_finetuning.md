# VQA Fine-Tuning Guide (Stage 2)

**Difficulty:** ⭐⭐⭐⭐ Advanced
**Time:** 40 minutes
**Config:** `uni_interleaved/configs/train/sft_vqa.yaml`

---

## Overview

Stage 2 fine-tunes the pretrained model on visual question answering datasets to improve instruction following and visual reasoning.

---

## Datasets

```yaml
datasets:
  - vqav2: 83k training samples
  - textvqa: 34k samples
  - okvqa: 9k samples
  - aokvqa: 17k samples
  - ocrvqa: 166k samples
  - gqa: 943k samples
  - llava: 665k instruction samples
```

**Total:** ~1.9M training samples

---

## Configuration

```yaml
model:
  freeze_llm: false       # Fine-tune full LLM
  freeze_vfm: true        # Freeze visual encoder
  freeze_dm: true         # Freeze image decoder

training:
  per_device_train_batch_size: 16
  learning_rate: 1e-5     # Lower than pretraining
  warmup_ratio: 0.03
  num_train_epochs: 3

  # Loss weights
  loss_txt_weight: 1.0
  loss_img_weight: 0.0    # No image generation in Stage 2
  loss_sniffer_weight: 0.0  # No diffusion feedback
```

---

## Dataset Format

### VQA Example

```json
{
  "image": "COCO_train2014_000000123456.jpg",
  "question": "What color is the cat?",
  "answer": "orange",
  "question_id": 123456
}
```

### Collated Sequence

```
<bos> <|soi|> <|image|>×77 Question: What color is the cat? Answer: orange <eos>
```

**Loss masking:** Only compute loss on "orange <eos>"

---

## Training Command

```bash
torchrun --nproc_per_node=8 \
    train.py \
    --config configs/train/sft_vqa.yaml \
    --output_dir ./OUTPUT/sft_vqa \
    --load_from_args ./OUTPUT/pretrain_stage1/checkpoint-100000 \
    --logging_steps 50 \
    --save_steps 2000
```

---

## Learning Rate Schedule

```python
# Warmup for 3% of steps
warmup_steps = total_steps * 0.03

# Linear decay
for step in range(total_steps):
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = max_lr * (1 - progress)
```

---

## Evaluation During Training

```yaml
evaluation_strategy: "steps"
eval_steps: 2000
```

**Metrics:**
- VQAv2 accuracy
- TextVQA accuracy
- GQA accuracy

**Expected progress:**
```
Step 0: VQAv2=62.0% (pretrained baseline)
Step 10000: VQAv2=64.5%
Step 20000: VQAv2=66.0%
Step 30000: VQAv2=66.8% (converged)
```

---

## Training Time

**8× A100 80GB:**
- Epoch 1: ~10 hours
- Epoch 2: ~10 hours
- Epoch 3: ~10 hours
- **Total:** ~30 hours

---

## Best Practices

### 1. Start from Good Checkpoint

```python
# Use best pretrain checkpoint
--load_from_args ./OUTPUT/pretrain_stage1/checkpoint-100000
```

### 2. Freeze What's Already Good

```yaml
freeze_vfm: true   # Visual encoder already learned
freeze_dm: true    # Don't need image generation
```

### 3. Use Lower Learning Rate

```yaml
learning_rate: 1e-5  # vs 1e-4 in pretraining
```

### 4. Augment Data

```python
# Random horizontal flip
random_flip: true

# Random resized crop
random_resize_crop_prob: 0.5
```

---

## Monitoring

```python
# Watch loss curves
tensorboard --logdir ./OUTPUT/sft_vqa/logs

# Key metrics:
# - train_loss: Should decrease to ~0.8
# - eval_vqa_accuracy: Should increase to ~67%
```

---

## Summary

**Duration:** ~30 hours on 8 GPUs
**Data:** VQA + instruction datasets
**Result:** VQAv2 ~66%, TextVQA ~52%

---

**Last Updated:** 2025-11-24
