# Text Generation Pipeline in DEEM

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes
**File:** `uni_interleaved/models/uni_interleaved.py:683-757`

---

## Overview

DEEM's text generation uses a cascaded architecture combining the multimodal LLM with a text decoder head. This tutorial explains the complete pipeline for VQA, captioning, and dialogue.

---

## Architecture

```
Image + Text Input
    ↓
[VisualTokenizer] → 77 visual tokens
    ↓
[Merge with text embeddings]
    ↓
[LLaMA with MMFS Cross-Attention]
    ↓
Hidden States [B, L, 4096]
    ↓
[Text Decoder LM Head]
    ↓
Token Logits [B, L, 32006]
    ↓
[Beam Search / Sampling]
    ↓
Generated Text
```

---

## Key Components

### 1. CascadeLlamaForCausalLMWrapper

**File:** `uni_interleaved/models/utils/causal_lm_cascade.py`

```python
class CascadeLlamaForCausalLMWrapper(nn.Module):
    def __init__(self, mm_decoder, text_decoder):
        self.mm_decoder = mm_decoder  # LlamaModel
        self.text_decoder = text_decoder  # TextDecoder (LM head)

    def forward(self, inputs_embeds, vision_hidden_states, ...):
        # Step 1: LLM forward
        mm_outputs = self.mm_decoder(
            inputs_embeds=inputs_embeds,
            vision_hidden_states=vision_hidden_states,
            cross_attention_mask=cross_attention_mask,
        )

        # Step 2: Text decoder
        text_outputs = self.text_decoder(
            inputs_embeds=mm_outputs.last_hidden_state
        )

        return text_outputs  # Logits [B, L, 32006]
```

**Purpose:** Wraps mm_decoder + text_decoder to be compatible with HuggingFace `generate()` API.

---

### 2. Generation Method

**File:** `uni_interleaved.py:683-757`

```python
def generate_texts(
    self,
    text_ids,
    image_tensors,
    num_image_per_seq,
    attention_mask,
    max_length=30,
    num_beams=5,
    temperature=1.0,
    use_nucleus_sampling=False,
    top_p=0.9,
    repetition_penalty=1.0,
    **kwargs
):
    # Step 1: Prepare multimodal embeddings
    _output = self._prepare_mm_embeds(
        text_ids=text_ids,
        image_tensors=image_tensors,
        image_tensors_dec=image_tensors_dec,
        num_image_per_seq=num_image_per_seq,
    )

    mm_embeds = _output["mm_embeds"]  # [B, L, 4096]
    cross_attention_mask = _output["cross_attention_mask"]
    mmfs_features_mm = _output["mmfs_features_mm"]

    # Step 2: Create wrapper for generation
    llm_wrapper = CascadeLlamaForCausalLMWrapper(
        self.mm_decoder,
        self.text_decoder,
    )

    # Step 3: Generate with HF API
    generate_text_ids = llm_wrapper.generate(
        inputs_embeds=mm_embeds,
        attention_mask=attention_mask,
        do_sample=use_nucleus_sampling,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_length,
        pad_token_id=31999,
        bos_token_id=1,
        eos_token_id=[2, 32000],  # Stop at <eos> or <|soi|>
        repetition_penalty=repetition_penalty,
        vision_hidden_states=mmfs_features_mm,
        cross_attention_mask=cross_attention_mask,
    )

    return {"text_ids": generate_text_ids}
```

---

## Generation Strategies

### Beam Search (Default)

**Parameters:**
```python
num_beams=5          # Keep top 5 hypotheses
length_penalty=1.0   # No length bias
```

**Behavior:** Deterministic, finds high-probability sequences

**Example:**
```python
outputs = model.generate_texts(
    ...,
    num_beams=5,
    max_length=50,
)
# Output: "A cat sitting on a mat in the living room."
```

---

### Nucleus Sampling (Creative)

**Parameters:**
```python
use_nucleus_sampling=True
top_p=0.9            # Sample from top 90% probability mass
temperature=1.0      # Randomness control
```

**Behavior:** Stochastic, more diverse outputs

**Example:**
```python
outputs = model.generate_texts(
    ...,
    use_nucleus_sampling=True,
    top_p=0.9,
    temperature=0.8,
)
# Output: "A fluffy cat relaxing on a cozy mat."  (varies each run)
```

---

## Stop Conditions

```python
eos_token_id=[
    2,      # <eos> - Normal completion
    32000,  # <|soi|> - Requesting image generation
]
```

**Behavior:**
- If model generates `<eos>`: Stop, return text
- If model generates `<|soi|>`: Stop, switch to image generation mode (interleaved)

---

## Example: Visual Question Answering

```python
from PIL import Image
import torch

# Load image
image = Image.open("cat.jpg")
image_tensor = transform(image)[0].unsqueeze(0).to("cuda")
image_tensor_dec = transform(image)[1].unsqueeze(0).to("cuda")

# Create VQA prompt
question = "What is the cat doing?"
text = f"<|soi|>{'<|image|>' * 77}{question}"

# Tokenize
tokens = tokenizer(text, return_tensors="pt")
text_ids = tokens.input_ids.to("cuda")
attention_mask = tokens.attention_mask.to("cuda")

# Generate answer
with torch.no_grad():
    outputs = model.generate_texts(
        text_ids=text_ids,
        image_tensors=image_tensor,
        image_tensors_dec=image_tensor_dec,
        num_image_per_seq=torch.tensor([1]),
        attention_mask=attention_mask,
        max_length=30,
        num_beams=3,
    )

# Decode
answer = tokenizer.decode(outputs["text_ids"][0], skip_special_tokens=True)
print(f"Q: {question}")
print(f"A: {answer}")
```

**Output:**
```
Q: What is the cat doing?
A: The cat is sitting on a mat.
```

---

## Example: Image Captioning

```python
# Captioning prompt (no question)
text = f"<|soi|>{'<|image|>' * 77}"

tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate_texts(
        text_ids=tokens.input_ids.to("cuda"),
        image_tensors=image_tensor,
        image_tensors_dec=image_tensor_dec,
        num_image_per_seq=torch.tensor([1]),
        attention_mask=tokens.attention_mask.to("cuda"),
        max_length=50,
        num_beams=5,
        use_nucleus_sampling=False,
    )

caption = tokenizer.decode(outputs["text_ids"][0], skip_special_tokens=True)
print(f"Caption: {caption}")
```

**Output:**
```
Caption: A fluffy orange cat sitting on a colorful woven mat in a sunny room.
```

---

## Hyperparameter Tuning

### For Accuracy (VQA)

```python
num_beams=5              # More beams
temperature=1.0          # No randomness
repetition_penalty=1.2   # Avoid repetition
max_length=20            # Short answers
```

### For Creativity (Captioning)

```python
use_nucleus_sampling=True
top_p=0.95
temperature=0.9
num_beams=1
max_length=100
```

### For Speed (Real-time)

```python
num_beams=1              # Greedy decoding
max_length=30
```

---

## Common Issues

### Issue 1: Repetitive Outputs

**Symptom:** "The cat the cat the cat..."

**Solution:**
```python
repetition_penalty=1.5   # Penalize repeated tokens
```

### Issue 2: Too Short Answers

**Symptom:** Single-word answers

**Solution:**
```python
min_length=10            # Minimum tokens
length_penalty=2.0       # Encourage longer outputs
```

### Issue 3: Generic Captions

**Symptom:** "A photo of an object"

**Solution:**
```python
use_nucleus_sampling=True
temperature=0.8
# Allows more diverse vocabulary
```

---

## Performance

**Throughput (A100 GPU):**
- Greedy: ~100 samples/second
- Beam search (5 beams): ~30 samples/second
- Nucleus sampling: ~80 samples/second

**Latency:**
- Average: 20-50ms per sample
- Max length 50: ~100ms

---

## Summary

**Pipeline:** Image → Visual Tokens → LLM → Text Decoder → Generated Text
**Key Methods:** Beam search, nucleus sampling
**Stop Tokens:** `<eos>` or `<|soi|>`
**Applications:** VQA, captioning, dialogue

---

## Next Steps

- [Image Generation Pipeline](06_image_generation_pipeline.md)
- [Interleaved Generation](07_interleaved_generation.md)
- [VQA Fine-Tuning](08_vqa_finetuning.md)

---

**Last Updated:** 2025-11-24
