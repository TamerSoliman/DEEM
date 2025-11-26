# LoRA Efficient Fine-Tuning

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 35 minutes

---

## Overview

LoRA (Low-Rank Adaptation) enables fine-tuning with <1% of parameters, reducing memory and time.

---

## Installation

```bash
pip install peft==0.3.0
```

---

## LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=[
        "q_proj",            # Query projections
        "v_proj",            # Value projections
        "k_proj",            # Key projections
        "o_proj",            # Output projections
        "llama_cross_attn",  # Cross-attention (DEEM-specific)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## Apply to Model

```python
model = MMInterleaved(**config.model)

# Wrap LLM component with LoRA
model.mm_decoder = get_peft_model(model.mm_decoder, lora_config)

# Print trainable parameters
model.mm_decoder.print_trainable_parameters()
# Output: trainable params: 42M || all params: 7B || trainable%: 0.6%
```

---

## Training

```yaml
# LoRA-specific settings
per_device_train_batch_size: 32  # 2x larger than full fine-tuning
learning_rate: 2e-4              # Higher LR for LoRA
gradient_accumulation_steps: 2
```

**Memory savings:** ~40% VRAM
**Speed:** ~1.5× faster

---

## Merge and Save

```python
# After training, merge LoRA weights
model.mm_decoder = model.mm_decoder.merge_and_unload()

# Save merged model
model.save_pretrained("./OUTPUT/lora_merged")
```

---

## Summary

**Benefits:** 40% less memory, 1.5× faster
**Performance:** ~98% of full fine-tuning accuracy
**Recommended for:** Limited GPU resources

---

**Last Updated:** 2025-11-24
