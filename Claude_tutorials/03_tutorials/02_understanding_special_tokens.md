# Understanding Special Tokens in DEEM

**Difficulty:** ⭐⭐ Novice
**Time:** 15 minutes
**Prerequisites:** Basic understanding of tokenization

---

## Overview

DEEM extends the Vicuna/LLaMA vocabulary with 6 special tokens for multimodal interactions. Understanding these tokens is crucial for working with the model.

---

## Special Token Dictionary

**Location:** `uni_interleaved/models/uni_interleaved.py:35-45`

```python
special_token_dict = dict(
    bos_token_id=1,          # Begin of sequence (LLaMA default)
    eos_token_id=2,          # End of sequence (LLaMA default)
    pad_token_id=31999,      # Padding token
    soi_token_id=32000,      # Start of image
    image_token_id=32001,    # Image placeholder (×77 per image)
    refleft_token_id=32002,  # Reference box left delimiter
    refright_token_id=32003, # Reference box right delimiter
    boxleft_token_id=32004,  # Bounding box left delimiter
    boxright_token_id=32005  # Bounding box right delimiter
)
```

### Vocabulary Extension

- **Original LLaMA:** 32,000 tokens (0-31,999)
- **DEEM Extended:** 32,006 tokens (+6 special tokens)

---

## Token Detailed Explanations

### 1. `<|soi|>` - Start of Image (ID: 32000)

**Purpose:** Marks the beginning of an image in the sequence

**Usage Pattern:**
```
<bos> Some text <|soi|> <|image|>×77 More text
                  ↑
            Image starts here
```

**Special Properties:**
- Has a **learnable embedding** (`self.soi_token` parameter)
- Added to base token embedding via `scatter_add` (not replace)
- Acts as boundary marker between text and visual content

**Code Reference:** `uni_interleaved.py:207-215`
```python
soi_token_pos = (text_ids == self.special_token_dict["soi_token_id"]).nonzero()
learnable_soi_embeds = self.soi_token.repeat(soi_token_pos.shape[0], 1)
mm_embeds = torch.scatter_add(mm_embeds, dim=0, index=soi_token_pos, src=learnable_soi_embeds)
```

---

### 2. `<|image|>` - Image Placeholder (ID: 32001)

**Purpose:** Placeholder tokens that get replaced with visual embeddings

**Usage Pattern:**
```
<|soi|> <|image|><|image|>...<|image|>
        ↑_________77 tokens________↑
```

**Key Facts:**
- **Exactly 77 tokens per image** (matches SD text encoder output length)
- Replaced during forward pass with visual embeddings from VisualTokenizer
- Never predicted during generation (masked out in loss)

**Why 77?**
- Stable Diffusion's text encoder outputs 77 tokens
- Perceiver Resampler compresses visual features to 77 tokens
- Standardized length for efficient batching

**Code Reference:** `uni_interleaved.py:191-205`
```python
image_token_pos = (text_ids == self.special_token_dict["image_token_id"]).nonzero()
mm_embeds = torch.scatter(text_embeds, dim=0, index=image_token_pos, src=valid_image_embeds)
```

---

### 3. `<refleft>` and `<refright>` - Reference Delimiters (IDs: 32002-32003)

**Purpose:** Delimit referring expressions (text that refers to image regions)

**Usage Pattern:**
```
Question: Where is <refleft> the cat <refright> ?
                    ↑___referring text___↑
```

**Use Cases:**
- Referring expression understanding
- Visual grounding tasks
- Part-based reasoning

**Example:**
```
Input: "Find <refleft> the person in red shirt <refright>"
Model: [Attends to specific image region matching description]
```

---

### 4. `<boxleft>` and `<boxright>` - Bounding Box Delimiters (IDs: 32004-32005)

**Purpose:** Delimit bounding box coordinates in normalized format

**Usage Pattern:**
```
The cat is at <boxleft> 0.25 0.30 0.65 0.80 <boxright>
                        ↑___x1  y1  x2  y2___↑
```

**Coordinate Format:**
- Normalized to [0, 1] range
- Format: `x1 y1 x2 y2` (top-left and bottom-right)
- Used for grounding tasks

**Example Sequence:**
```
<bos> <|soi|> <|image|>×77 The cat is at <boxleft> 0.3 0.4 0.7 0.8 <boxright> <eos>
```

---

## Token Initialization

**Location:** `uni_interleaved/custom_datasets/utils/wds_utils.py`

```python
def init_tokenizer(tokenizer_path):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '<|soi|>',       # ID 32000
            '<|image|>',     # ID 32001
            '<refleft>',     # ID 32002
            '<refright>',    # ID 32003
            '<boxleft>',     # ID 32004
            '<boxright>',    # ID 32005
        ]
    })
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer
```

**Embedding Extension:**
```python
# In MMInterleaved.__init__
llm_model = LlamaForCausalLM.from_pretrained(llm_model_path)
orig_txt_vocab_size = llm_model.config.vocab_size  # 32000
llm_model.resize_token_embeddings(txt_vocab_size)   # 32006
```

---

## Common Usage Patterns

### Pattern 1: Visual Question Answering

```python
# Input format
text = "<bos> <|soi|> " + "<|image|>" * 77 + " What is in this image? "

# Model generates
output = "A cat sitting on a mat. <eos>"
```

### Pattern 2: Image Generation

```python
# Input format
text = "<bos> A beautiful sunset over mountains <|soi|> " + "<|image|>" * 77

# Model generates image at <|image|> positions
```

### Pattern 3: Interleaved Generation

```python
# Text → Image → Text
text = "<bos> Here is a cat <|soi|> " + "<|image|>" * 77 + " It is chasing <|soi|> " + "<|image|>" * 77

# Generates:
# - Image 1 from "Here is a cat"
# - Image 2 from "Here is a cat [img1] It is chasing"
```

### Pattern 4: Grounding with Boxes

```python
text = "<bos> <|soi|> " + "<|image|>" * 77 + " The cat is at <boxleft> 0.3 0.4 0.7 0.8 <boxright>"
```

---

## Token Masking in Loss Computation

**Location:** `uni_interleaved.py:388-452`

### Which tokens are ignored in loss?

```python
# 1. Ignore <|image|> tokens (visual embeddings, not text)
gt_text_ids.masked_fill_(text_ids == image_token_id, -100)

# 2. Ignore padding tokens
gt_text_ids.masked_fill_(text_ids == pad_token_id, -100)

# 3. Ignore <bos> tokens
gt_text_ids.masked_fill_(text_ids == bos_token_id, -100)

# 4. Ignore <bos> → <|soi|> transitions
is_bos2soi = (text_ids[:, :-1] == bos_token_id) & (text_ids[:, 1:] == soi_token_id)
gt_text_ids.masked_fill_(is_bos2soi, -100)
```

### Why mask these?

- `<|image|>`: Visual placeholders, not language tokens
- `<bos>`, `<pad>`: Structural tokens, not content
- `<bos> → <|soi|>`: Deterministic transition, no need to learn

---

## Token Usage in Generation

### Text Generation

```python
# Model stops at:
eos_token_id = [
    self.special_token_dict["eos_token_id"],  # 2
    self.special_token_dict["soi_token_id"],  # 32000 (start image gen)
]
```

**Behavior:**
- If model generates `<eos>`: Stop completely
- If model generates `<|soi|>`: Switch to image generation mode

### Image Generation Detection

**Location:** `inference.py:169-181`

```python
# Check if last token is <|soi|>
if text_ids[-1] == special_token_dict["soi_token_id"]:
    # Append 77 <|image|> tokens
    image_ids = [image_token_id] * 77
    text_ids = torch.cat((text_ids, image_ids), dim=-1)

    # Add placeholder image tensor
    inputs["image_tensors"] = torch.cat([inputs["image_tensors"], pad_image_tensor])

    # Switch mode
    gen_image_next = True
```

---

## Practical Examples

### Example 1: Create VQA Input

```python
from PIL import Image

image = Image.open("cat.jpg")
question = "What animal is this?"

# Format with special tokens
text = f"<|soi|>{'<|image|>' * 77}{question}"

# Tokenize
tokens = tokenizer(text, return_tensors="pt")
print(tokens.input_ids[0, :85])  # First 85 tokens

# Output:
# [1,        # <bos>
#  32000,    # <|soi|>
#  32001, 32001, ..., 32001,  # <|image|> ×77
#  1724, 13019, 338, 445]     # "What animal is this"
```

### Example 2: Parse Generated Text with Boxes

```python
output_text = "The dog is at <boxleft> 0.2 0.3 0.6 0.7 <boxright>"

# Extract box coordinates
import re
box_pattern = r'<boxleft>\s*([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s*<boxright>'
match = re.search(box_pattern, output_text)

if match:
    x1, y1, x2, y2 = map(float, match.groups())
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
    # Output: Bounding box: (0.2, 0.3, 0.6, 0.7)
```

### Example 3: Multi-Image Sequence

```python
# Format for 3 interleaved images
num_images = 3
text_parts = [
    "A cat",
    "is chasing",
    "a mouse"
]

sequence = "<bos>"
for i in range(num_images):
    sequence += f" {text_parts[i]} <|soi|> " + "<|image|>" * 77

print(f"Total tokens: ~{1 + num_images * (77 + 5)}")  # ~247 tokens
```

---

## Debugging Token Issues

### Issue 1: Wrong Number of Image Tokens

**Error:** `AssertionError: image_token_pos.shape != valid_image_embeds.shape`

**Cause:** Number of `<|image|>` tokens doesn't match images

**Solution:**
```python
# Always use exactly 77 tokens per image
num_images = 2
text = "<bos> " + ("<|soi|> " + "<|image|>" * 77) * num_images
```

### Issue 2: Token ID Out of Range

**Error:** `RuntimeError: index out of range`

**Cause:** Model not resized for extended vocabulary

**Solution:**
```python
# Ensure model vocabulary is extended
assert model.mm_decoder.config.vocab_size == 32006
assert model.text_decoder.txt_vocab_size == 32006
```

### Issue 3: Generation Doesn't Stop

**Cause:** `<eos>` token not in stop criteria

**Solution:**
```python
# Include both stop tokens
outputs = model.generate_texts(
    ...,
    eos_token_id=[2, 32000],  # <eos> and <|soi|>
)
```

---

## Token Embeddings

### Learnable vs. Fixed

| Token | Embedding Type | Learned? |
|-------|----------------|----------|
| `<bos>`, `<eos>`, `<pad>` | LLaMA default | ✅ During LLM pretraining |
| `<|image|>` | Random init → **Replaced** | ❌ Gets replaced by visual embeddings |
| `<|soi|>` | Random init + **Learnable param** | ✅ `self.soi_token` learned |
| `<refleft>`, `<refright>` | Random init | ✅ During fine-tuning |
| `<boxleft>`, `<boxright>` | Random init | ✅ During fine-tuning |

### Special: `<|soi|>` Dual Embedding

```python
# Two embeddings combined:
# 1. Token embedding (like regular tokens)
text_embed = embedding_layer(32000)

# 2. Learnable parameter (task-specific)
soi_param = self.soi_token  # nn.Parameter(torch.zeros(1, 4096))

# Final embedding (scatter_add, not scatter)
final_embed = text_embed + soi_param
```

**Why dual?**
- Base embedding: Shares information with text tokens
- Learnable param: Task-specific boundary signal

---

## Best Practices

### ✅ Do's

1. **Always use 77 `<|image|>` tokens per image**
2. **Place `<|soi|>` before image tokens**
3. **Include `<bos>` at sequence start**
4. **Normalize box coordinates to [0, 1]**
5. **Use `skip_special_tokens=True` when decoding for display**

### ❌ Don'ts

1. **Don't mix `<|image|>` counts** (must be exactly 77)
2. **Don't put text between `<|soi|>` and `<|image|>` tokens**
3. **Don't predict `<|image|>` tokens** (mask in loss)
4. **Don't forget to resize embeddings** when loading pretrained model

---

## Summary

### Token ID Quick Reference

```
Standard LLaMA:
0-31,999  : Vocabulary tokens
1         : <bos>
2         : <eos>
31,999    : <pad>

DEEM Extensions:
32,000    : <|soi|>        (Start of Image)
32,001    : <|image|>      (Image Placeholder ×77)
32,002    : <refleft>      (Reference Left)
32,003    : <refright>     (Reference Right)
32,004    : <boxleft>      (Box Left)
32,005    : <boxright>     (Box Right)
```

### Usage Formulas

```python
# VQA
f"<|soi|>{'<|image|>' * 77}{question}"

# Image Generation
f"{prompt}<|soi|>{'<|image|>' * 77}"

# Grounding
f"<|soi|>{'<|image|>' * 77} Object at <boxleft> {x1} {y1} {x2} {y2} <boxright>"

# Interleaved
f"{text1}<|soi|>{'<|image|>' * 77} {text2}<|soi|>{'<|image|>' * 77}"
```

---

## Next Steps

- [Visual Tokenizer Deep Dive](04_visual_tokenizer.md) - How images become 77 tokens
- [Text Generation Pipeline](07_text_generation.md) - How special tokens control generation
- [Annotated Forward Pass](../02_annotated_code/ANNOTATED_FORWARD_PASS.md) - See token handling in code

---

**Last Updated:** 2025-11-24
