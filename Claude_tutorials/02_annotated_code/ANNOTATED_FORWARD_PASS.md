# Annotated Multimodal Forward Pass

This document provides a **heavily annotated, line-by-line explanation** of the DEEM multimodal forward pass, explaining the **What, Where, When, Why, How, and flow context** of every major operation.

**Source File:** `uni_interleaved/models/uni_interleaved.py:454-590`

---

## 📚 Table of Contents

1. [Function Signature & Inputs](#function-signature--inputs)
2. [Step 1: Initialization](#step-1-initialization)
3. [Step 2: Prepare Multimodal Embeddings](#step-2-prepare-multimodal-embeddings)
4. [Step 3: LLM Forward Pass with Cross-Attention](#step-3-llm-forward-pass-with-cross-attention)
5. [Step 4: Text Decoder Forward](#step-4-text-decoder-forward)
6. [Step 5: Compute Text Loss](#step-5-compute-text-loss)
7. [Step 6: Image Decoder Forward](#step-6-image-decoder-forward)
8. [Step 7: Return Combined Loss](#step-7-return-combined-loss)
9. [Complete Flow Diagram](#complete-flow-diagram)

---

## Function Signature & Inputs

```python
def forward(
    self,
    text_ids: torch.LongTensor,              # [B, L] Token IDs
    image_tensors: Optional[torch.FloatTensor] = None,        # [N_img, 3, 224, 224]
    image_tensors_mask: Optional[torch.FloatTensor] = None,   # [N_img, 1, 224, 224]
    image_tensors_dec: Optional[torch.FloatTensor] = None,    # [N_img, 3, 512, 512]
    num_image_per_seq: Optional[torch.Tensor] = None,         # [B]
    attention_mask: Optional[torch.Tensor] = None,            # [B, L]
    gt_text_ids: Optional[torch.LongTensor] = None,           # [B, L-1] (optional)
    nearest_bos_idxs: Optional[torch.Tensor] = None,          # [N_img]
    ignore_prompt_token_offset=0,                              # int or List[int]
    loss_img_weight=None,                                      # float (override)
    loss_txt_weight=None,                                      # float (override)
    loss_sniffer_weight=None,                                  # float (override)
    meta=None,                                                 # dict (metadata)
    image_loss_mask=None,                                      # [N_img, 1, H//8, W//8]
    **kwargs,
):
```

### 🔍 Input Breakdown

| Parameter | Shape | Description | Example |
|-----------|-------|-------------|---------|
| `text_ids` | `[B, L]` | Tokenized text with `<|image|>` placeholders | `[1, <bos>, <|soi|>, <|image|>×77, "cat", ...]` |
| `image_tensors` | `[N_img, 3, 224, 224]` | Images for CLIP encoder | Preprocessed, normalized |
| `image_tensors_dec` | `[N_img, 3, 512, 512]` | Images for SD decoder | Higher resolution |
| `image_tensors_mask` | `[N_img, 1, H, W]` | Binary masks for regions | For referring tasks |
| `num_image_per_seq` | `[B]` | Images per batch item | `[2, 1, 3]` → 3 samples with 2, 1, 3 images |
| `attention_mask` | `[B, L]` | Token attention mask | `1` for valid tokens, `0` for padding |
| `gt_text_ids` | `[B, L-1]` | Ground truth tokens (shifted) | For custom loss masking |
| `nearest_bos_idxs` | `[N_img]` | Nearest `<bos>` index per image | For context extraction |
| `meta` | `dict` | Metadata | `{"dataset_name": "vqav2", ...}` |

### 📌 Key Concepts

**What is `<|image|>` token?**
- Special token (ID: 32001) that serves as a placeholder for visual embeddings
- Each image is represented by **77 consecutive `<|image|>` tokens**
- During forward pass, these tokens are replaced with actual visual embeddings from the VisualTokenizer

**Example Text Sequence:**
```
<bos> <|soi|> <|image|>×77 What is in this image? <eos>
```
- `<bos>` (ID: 1): Begin of sequence
- `<|soi|>` (ID: 32000): Start of image marker
- `<|image|>×77` (ID: 32001): 77 visual embedding placeholders
- Text tokens: "What is in this image?"
- `<eos>` (ID: 2): End of sequence

---

## Step 1: Initialization

### 📍 Location: Lines 472-475

```python
output, loss = {}, 0.0

if image_tensors_mask is None:
    image_tensors_mask=torch.ones_like(image_tensors)[:,0:1,...]
```

### 🔍 What's Happening

- **Initialize output dictionary** and total loss accumulator
- **Create default mask** if not provided (all pixels valid)

### ❓ Why?

- **Mask Flexibility:** Some tasks (VQA, caption) don't use masks; referring tasks do
- **Default Behavior:** If no mask, treat entire image as region of interest

### 💡 Proximal Context

- **Before:** Function just started
- **After:** Ready to process inputs

---

## Step 2: Prepare Multimodal Embeddings

### 📍 Location: Lines 477-501

```python
_output = self._prepare_mm_embeds(
    text_ids=text_ids,
    image_tensors=image_tensors,
    image_tensors_dec=image_tensors_dec,
    image_tensors_mask=image_tensors_mask,
    num_image_per_seq=num_image_per_seq,
    meta=meta,
)
```

### 🔍 What's Happening

This calls the **most critical preprocessing function** that:
1. Encodes images via VisualTokenizer
2. Replaces `<|image|>` tokens with actual visual embeddings
3. Prepares cross-attention masks and multi-scale features

### 🎯 Detailed Breakdown: `_prepare_mm_embeds()`

#### **Step 2.1: Get Text Embeddings**
**Location:** `uni_interleaved.py:170`

```python
text_embeds = self.mm_decoder.get_input_embeddings()(text_ids)
B, L, C = text_embeds.shape  # Batch, Length, Channels (hidden_size)
```

**What:** Convert token IDs → embeddings using LLaMA's embedding layer

**Example:**
```python
text_ids = [1, 32000, 32001, ..., 32001, 1234, ...]  # Token IDs
text_embeds = [[0.1, 0.5, ...], [0.2, 0.3, ...], ...]  # [B, L, 4096]
```

---

#### **Step 2.2: Encode Images with VisualTokenizer**
**Location:** `uni_interleaved.py:179-184`

```python
visual_output = self.visual_tokenizer(image_tensors, image_tensors_dec, image_tensors_mask)

valid_image_embeds = visual_output["vis_embed"]  # [N_img, 77, C]
output["loss_sniffer"] = visual_output["loss_sniffer"]
valid_image_embeds = rearrange(valid_image_embeds, "b l c -> (b l) c")  # [N_img*77, C]
```

**What:**
1. `visual_tokenizer.forward()` encodes images
2. Returns **77 visual tokens per image**
3. Also returns **multi-scale features** and **diffusion feedback loss**
4. Flatten to `[N_img*77, C]` for scatter operation

**Visual Tokenizer Process:**
```
Image (3×224×224)
    ↓ [CLIP ViT/ConvNeXT]
Multi-scale Features (257 tokens)
    ↓ [MaskPooling + Position Embeddings]
Region-Aware Features
    ↓ [Perceiver Resampler]
77 Compressed Tokens
    ↓ [Projection to LLM Hidden Size]
vis_embed: [B, 77, 4096]
```

**Loss Sniffer:**
```python
# If diffusion feedback is enabled:
diffusion_loss = SD_encoder(image_dec, vis_embed)
sniffer_loss = diffusion_loss * mask_region  # Focus on masked regions
```

---

#### **Step 2.3: Insert Visual Embeddings into Text Sequence**
**Location:** `uni_interleaved.py:191-216`

```python
# Find positions of <|image|> tokens (ID: 32001)
image_token_pos_x, image_token_pos_y = (
    text_ids == self.special_token_dict["image_token_id"]
).nonzero(as_tuple=True)
```

**What:** Locate all `<|image|>` tokens in the flattened sequence

**Example:**
```python
text_ids = [[1, 32000, 32001, 32001, ..., 32001, 1234, 5678, 2]]
                          ↑      ↑           ↑
                         pos=2  pos=3      pos=78
```

```python
# Convert 2D positions to 1D flattened indices
image_token_pos = image_token_pos_x * L + image_token_pos_y
```

**Example:**
```python
# Batch 0, position 2 → flattened index = 0*2048 + 2 = 2
# Batch 0, position 3 → flattened index = 0*2048 + 3 = 3
# ...
```

```python
# Flatten text embeddings
text_embeds = rearrange(text_embeds, "b l c -> (b l) c")  # [B*L, C]

# Scatter visual embeddings into text positions
image_token_pos = image_token_pos[:, None].expand(-1, C)  # [N_img*77, C]
mm_embeds = torch.scatter(
    text_embeds, dim=0, index=image_token_pos, src=valid_image_embeds
)
```

**What:** Replace `<|image|>` token embeddings with actual visual embeddings

**Visual Explanation:**
```
BEFORE:
text_embeds = [emb(1), emb(32000), emb(32001), emb(32001), ..., emb(1234)]

AFTER (scatter operation):
mm_embeds = [emb(1), emb(32000), vis_emb_0, vis_emb_1, ..., emb(1234)]
```

---

#### **Step 2.4: Add Learnable Start-of-Image Token**
**Location:** `uni_interleaved.py:207-216`

```python
# Find <|soi|> tokens (ID: 32000)
soi_token_pos_x, soi_token_pos_y = (
    text_ids == self.special_token_dict["soi_token_id"]
).nonzero(as_tuple=True)
soi_token_pos = soi_token_pos_x * L + soi_token_pos_y

# Add learnable embedding to <|soi|> positions
learnable_soi_embeds = self.soi_token.repeat(soi_token_pos.shape[0], 1)
mm_embeds = torch.scatter_add(
    mm_embeds, dim=0, index=soi_token_pos, src=learnable_soi_embeds
)
```

**Why `scatter_add` instead of `scatter`?**
- `scatter`: **Replaces** the embedding
- `scatter_add`: **Adds** to existing embedding
- **Rationale:** Keep original `<|soi|>` embedding + add learnable signal

**Purpose of `soi_token`:**
- Learnable parameter: `nn.Parameter(torch.zeros(1, 4096))`
- Acts as a **boundary marker** between text and image
- Helps model learn when to transition from reading text to processing image

---

#### **Step 2.5: Reshape Back to Batch Format**
**Location:** `uni_interleaved.py:216-217`

```python
mm_embeds = rearrange(mm_embeds, "(b l) c -> b l c", b=B)
output["mm_embeds"] = mm_embeds  # [B, L, C]
```

---

#### **Step 2.6: Prepare Cross-Attention Masks and MMFS Features**
**Location:** `uni_interleaved.py:220-227`

```python
output.update(
    self._prepare_mmfs_features_for_mm_decoder(
        text_ids,
        num_image_per_seq,
        visual_output["multiscale_features"],
    )
)
output["multiscale_features"] = visual_output["multiscale_features"]
```

**What:** Prepares data for **MMFS (Multi-Image Multi-Scale Feature Synchronizer)**

### 🔍 Deep Dive: `_prepare_mmfs_features_for_mm_decoder()`

**Location:** `uni_interleaved.py:231-298`

#### **Step 2.6.1: Build Cross-Attention Mask**

```python
# Find all <|soi|> token positions
soi_token_pos = (text_ids == self.special_token_dict["soi_token_id"]).nonzero()[1]

# Create image position tensor [B, max_num_image]
image_token_pos = -1 * torch.ones(B, max_num_image).type_as(soi_token_pos)

# Populate with actual positions
for i in range(B):
    image_token_pos[i, : num_image_per_seq[i]] = (
        soi_token_pos[start_idx : start_idx + num_image_per_seq[i]] + 1
    )
    start_idx += num_image_per_seq[i]
```

**What:** For each batch item, record where each image starts (`<|soi|> + 1`)

**Example:**
```python
text_ids[0] = [<bos>, <|soi|>, <|image|>×77, "cat", <|soi|>, <|image|>×77, ...]
                      ↑ pos=1                         ↑ pos=80

image_token_pos[0] = [2, 81, -1, -1, ...]  # Starts of image embeddings
```

```python
# Create attention mask: which images can attend to which tokens?
attention_mask = (
    (image_token_pos > nearest_bos_ids[:, None, :])   # After last <bos>
    * (image_token_pos <= index)                       # Before or at current position
    * (image_token_pos != -1)                          # Valid image
)  # [B, N_images, L]

attention_mask = attention_mask.transpose(-1, -2).float()  # [B, L, N_images]
output["cross_attention_mask"] = attention_mask
```

**What:** Create causal mask for cross-attention

**Visual Explanation:**
```
Text: [<bos>, <|soi|>, <img>×77, "cat", <|soi|>, <img>×77, "sits"]
Pos:  [0,     1,       2-78,    79,    80,      81-157,   158]

Cross-Attention Mask [L, N_images]:
        Image0  Image1
Pos 0:    0       0     (can't see any images yet)
Pos 1:    0       0     (<|soi|> marker, no image yet)
Pos 2:    1       0     (first image token can see image 0)
...
Pos 79:   1       0     ("cat" can see image 0)
Pos 80:   1       0     (<|soi|> for image 1, can see image 0)
Pos 81:   1       1     (first token of image 1, can see both)
...
Pos 158:  1       1     ("sits" can see both images)
```

---

#### **Step 2.6.2: Prepare Multi-Scale Features for MMFS**

```python
# Extract features at specified spatial scales
mmfs_features = []
for feat in multiscale_features:
    shape = int(feat.shape[-1])
    if shape in self.spatial_shapes:  # [32, 16, 8]
        mmfs_features.append(feat)
```

**What:** Filter multi-scale features to match configured spatial shapes

**Example:**
```python
multiscale_features from ViT:
- Stage 1: [N_img, 1024, 64, 64]  # Not used
- Stage 2: [N_img, 1024, 32, 32]  # ✓ Used
- Stage 3: [N_img, 1024, 16, 16]  # ✓ Used
- Stage 4: [N_img, 1024, 8, 8]    # ✓ Used
```

```python
# Reshape to [B, max_num_image, C, H, W]
mmfs_features_new = [
    torch.zeros(B, max_num_image, *feat.shape[1:], ...)
    for feat in mmfs_features
]

# Populate with actual features
for i in range(B):
    item = feat[start_idx : start_idx + num_image_per_seq[i]]
    feat_n[i, : item.shape[0], ...] = item
```

**Why reshape?**
- **Batching:** Different samples may have different numbers of images
- **Padding:** Pad to `max_num_image` for efficient batching
- **MMFS Input:** MMFS expects `[B, N_images, HW, C]` format

```python
# Flatten spatial dimensions
mmfs_features_mm = []
for feat in mmfs_features_new:
    feat_n = rearrange(feat, "b n c h w -> b n (h w) c")  # [B, N, HW, C]
    mmfs_features_mm.append(feat_n)

# Concatenate all scales
mmfs_features_mm = torch.cat(mmfs_features_mm, dim=2)  # [B, N, HW1+HW2+HW3, C]
```

**Final Shape:**
```python
# 32×32 + 16×16 + 8×8 = 1024 + 256 + 64 = 1344 spatial positions
mmfs_features_mm: [B, max_num_image, 1344, 1024]
```

---

### 📤 Output of `_prepare_mm_embeds()`

```python
return {
    "mm_embeds": mm_embeds,                      # [B, L, C]
    "cross_attention_mask": attention_mask,      # [B, L, N_images]
    "mmfs_features_mm": mmfs_features_mm,        # [B, N_images, 1344, C]
    "multiscale_features": multiscale_features,  # List of [N_img, C, H, W]
    "loss_sniffer": sniffer_loss                 # Scalar
}
```

---

### 📍 Back to Main Forward Pass: Lines 486-500

```python
# Extract outputs
mm_embeds = _output.pop("mm_embeds")                    # [B, L, C]
cross_attention_mask = _output.pop("cross_attention_mask", None)  # [B, L, N]
mmfs_features_mm = _output.pop("mmfs_features_mm", None)          # [B, N, HW, C]
loss_sniffer = _output.pop("loss_sniffer")

# Accumulate sniffer loss
loss_sniffer_weight = (
    loss_sniffer_weight if loss_sniffer_weight is not None else self.loss_sniffer_weight
)
loss = loss + loss_sniffer.mean() * loss_sniffer_weight  # Default: 5.0

output["loss_sniffer"] = loss_sniffer.mean().detach()
output.update(_output)  # Save multiscale_features
```

**What:**
- Extract prepared embeddings and features
- Compute **diffusion feedback loss** weighted by `loss_sniffer_weight` (default: 5.0)
- Store for logging

**Why Sniffer Loss?**
- Forces visual encoder to learn features aligned with diffusion model
- Acts as a **perceptual regularizer**
- Helps with visual understanding quality

---

## Step 3: LLM Forward Pass with Cross-Attention

### 📍 Location: Lines 503-513

```python
# Enable gradients for multimodal embeddings
mm_embeds.requires_grad_(True)

# Forward through LLaMA with cross-attention
mm_outputs = self.mm_decoder(
    inputs_embeds=mm_embeds,              # [B, L, C] instead of input_ids
    attention_mask=attention_mask,        # [B, L] standard attention
    vision_hidden_states=mmfs_features_mm,  # [B, N, HW, C] visual features
    cross_attention_mask=cross_attention_mask,  # [B, L, N] cross-attention
    return_dict=True,
    output_hidden_states=True,
)
mm_hidden_state = mm_outputs.last_hidden_state  # [B, L, C]
```

### 🔍 What's Happening

**LLaMA Forward with Modifications:**

1. **Standard Self-Attention Layers (Most Layers):**
   ```python
   # Normal transformer layer
   hidden_states = self_attn(hidden_states, attention_mask)
   hidden_states = ffn(hidden_states)
   ```

2. **Cross-Attention Layers (Every 4th Layer):**
   ```python
   # First do self-attention
   hidden_states = self_attn(hidden_states, attention_mask)

   # Then cross-attention with visual features (MMFS)
   hidden_states = llama_cross_attn(
       hidden_states,              # Query: text features
       vision_hidden_states,       # Key/Value: multi-scale image features
       cross_attention_mask        # Mask: which images each token can see
   )

   # Then FFN
   hidden_states = ffn(hidden_states)
   ```

**Example Flow:**
```
Layer 0:  Self-Attn → FFN
Layer 1:  Self-Attn → FFN
Layer 2:  Self-Attn → FFN
Layer 3:  Self-Attn → FFN
Layer 4:  Self-Attn → Cross-Attn (MMFS) → FFN  ← Visual injection
Layer 5:  Self-Attn → FFN
...
Layer 8:  Self-Attn → Cross-Attn (MMFS) → FFN  ← Visual injection
...
Layer 31: Self-Attn → FFN
```

### 🎯 MMFS Cross-Attention Details

**Query:** Text hidden states `[B, L, 4096]`
**Key/Value:** Multi-scale image features `[B, N_images, 1344, 1024]`

```python
# MMFS computes:
for each text token at position l:
    for each image n that token l can see (via cross_attention_mask):
        # Sample from multi-scale features using deformable attention
        sampled_features = deformable_sample(
            image_features[n],  # [1344, 1024]
            spatial_levels=[32, 16, 8],
            num_heads=16,
            num_points=8  # Sample 8 points per scale
        )
        # Aggregate with attention weights
        aggregated[l, n] = attention_weights @ sampled_features

    # Combine across all visible images
    output[l] = combine(aggregated[l, :])
```

**Why MMFS?**
- **Multi-Scale:** Captures both fine details (8×8) and global context (32×32)
- **Multi-Image:** Handles multiple interleaved images efficiently
- **Deformable:** Learns where to look in the image (attention offsets)

---

## Step 4: Text Decoder Forward

### 📍 Location: Lines 515-522

```python
# Clone hidden states for text decoder
mm_hidden_state_txt = mm_hidden_state.clone()

# Forward through text decoder (LM head)
text_decode_outputs = self.text_decoder(
    inputs_embeds=mm_hidden_state_txt,  # [B, L, C]
    attention_mask=attention_mask,       # [B, L]
    return_dict=True,
)
text_logits = text_decode_outputs.logits  # [B, L, 32006]
text_logits = rearrange(text_logits, "b n c -> b c n")  # [B, 32006, L]
```

### 🔍 What's Happening

**Text Decoder = LM Head:**
```python
class TextDecoder:
    def forward(self, inputs_embeds):
        # Linear projection: hidden_size → vocab_size
        logits = self.lm_head(inputs_embeds)  # [B, L, 4096] → [B, L, 32006]
        return logits
```

**Vocabulary Size:**
- Original LLaMA: 32,000 tokens
- Extended: **32,006 tokens** (+6 special tokens)

**Special Tokens:**
```
32000: <|soi|>      (start of image)
32001: <|image|>    (image placeholder)
32002: <refleft>    (reference box start)
32003: <refright>   (reference box end)
32004: <boxleft>    (bbox coordinate start)
32005: <boxright>   (bbox coordinate end)
```

---

## Step 5: Compute Text Loss

### 📍 Location: Lines 524-542

```python
# Prepare ground truth labels
gt_text_ids = self._prepare_gt_text_ids(
    text_ids,
    attention_mask=attention_mask,
    ignore_prompt_token_offset=ignore_prompt_token_offset,
    gt_text_ids=gt_text_ids,
    meta=meta,
)  # [B, L-1]
```

### 🔍 Deep Dive: `_prepare_gt_text_ids()`

**Location:** `uni_interleaved.py:388-452`

#### **Step 5.1: Basic Masking**

```python
# Clone and shift by 1 (next-token prediction)
gt_text_ids = text_ids.clone()

# Ignore prompt tokens (don't compute loss on prompt)
if isinstance(ignore_prompt_token_offset, int):
    gt_text_ids[:, :ignore_prompt_token_offset] = -100
else:
    # Per-sample offsets
    for idx, offset in enumerate(ignore_prompt_token_offset):
        gt_text_ids[idx, :offset] = -100
```

**Example:**
```python
text_ids = [<bos>, <|soi|>, <|image|>×77, "What", "is", "this", "?", "A", "cat", <eos>]
ignore_offset = 84  # Don't compute loss on prompt (up to "?")

gt_text_ids = [-100, -100, ..., -100, "A", "cat", <eos>]
```

---

#### **Step 5.2: Ignore Image-Only Context Loss** (Optional)

```python
# For some datasets (e.g., MMC4), ignore loss on text before first image
if meta["dataset_name"] in self.dataset_to_ignore_noimage_cond_loss:
    # Find nearest <|soi|> for each position
    nearest_soi_idxs = ...

    # Mask tokens before any image
    noimage_cond_token = (nearest_soi_idxs < nearest_bos_idxs) or (nearest_soi_idxs == -1)
    gt_text_ids = gt_text_ids.masked_fill(noimage_cond_token, -100)
```

**Why?**
- Some datasets have text before images: "Caption: <image>"
- We want model to generate captions **conditioned on image**, not just text

---

#### **Step 5.3: Shift and Mask Special Tokens**

```python
# Shift by 1 for next-token prediction
gt_text_ids = gt_text_ids[:, 1:]  # [B, L-1]

# Ignore padding tokens
gt_text_ids = gt_text_ids.masked_fill(
    text_ids[:, 1:] == self.special_token_dict["pad_token_id"], -100
)

# Ignore <|image|> tokens (don't predict image tokens)
gt_text_ids = gt_text_ids.masked_fill(
    text_ids[:, 1:] == self.special_token_dict["image_token_id"], -100
)

# Ignore attention-masked positions
gt_text_ids = gt_text_ids.masked_fill(attention_mask[:, 1:] == 0, -100)

# Ignore <bos> → <|soi|> transitions
is_bos_token = text_ids[:, :-1] == self.special_token_dict["bos_token_id"]
is_soi_token = text_ids[:, 1:] == self.special_token_dict["soi_token_id"]
is_bos2soi_token = torch.logical_and(is_bos_token, is_soi_token)
gt_text_ids = gt_text_ids.masked_fill(is_bos2soi_token, -100)

# Ignore <bos> tokens themselves
gt_text_ids = gt_text_ids.masked_fill(
    text_ids[:, 1:] == self.special_token_dict["bos_token_id"], -100
)
```

**Result:** `gt_text_ids` with `-100` for positions to ignore in loss

---

### 📍 Back to Loss Computation: Lines 532-542

```python
# Compute cross-entropy loss
text_logits = text_logits.float()  # [B, 32006, L]
loss_txt = F.cross_entropy(
    text_logits[..., :-1].contiguous(),  # [B, 32006, L-1]
    gt_text_ids.contiguous(),             # [B, L-1]
    reduction="mean",
)

# Weight and accumulate
loss_txt_weight = (
    loss_txt_weight if loss_txt_weight is not None else self.loss_txt_weight
)  # Default: 1.0
loss = loss + loss_txt * loss_txt_weight

output["loss_txt"] = loss_txt.detach()
```

**What:**
- Standard language modeling loss (cross-entropy)
- Ignores positions with `-100` in `gt_text_ids`
- Weighted by `loss_txt_weight` (default: 1.0)

---

## Step 6: Image Decoder Forward

### 📍 Location: Lines 544-587

```python
if self.image_decoder is not None:
    # Clone hidden states for image decoder
    mm_hidden_state_img = mm_hidden_state.clone()
    context_features = mm_hidden_state_img
```

**When is `image_decoder` None?**
- Stage 2 (VQA-only training): No image generation, only text
- Image decoder is only used when `image_decoder_config` is provided

---

### **Step 6.1: Prepare Context Features**

**Location:** Lines 550-558

```python
(
    context_features,
    context_attention_mask,
) = self._prepare_context_features_for_image_decoder(
    context_features,
    text_ids=text_ids,
    image_start_token_idx=None,  # Auto-detect from <|soi|>
    nearest_bos_idxs=nearest_bos_idxs,
)
```

### 🔍 Deep Dive: `_prepare_context_features_for_image_decoder()`

**Location:** `uni_interleaved.py:300-350`

```python
# Find all <|soi|> positions
image_start_token_idx = (
    text_ids == self.special_token_dict["soi_token_id"]
).nonzero(as_tuple=True)[-1]  # Column indices

# For each image, extract context from [nearest_bos : soi+1]
for i in range(N_images):
    row_idx = image_start_token_row_ids[i]
    _context_features = context_features[
        row_idx, nearest_bos_idxs[i] : image_start_token_idx[i] + 1, :
    ]
    # Reverse order (put <|soi|> first)
    _context_features = _context_features.flip(dims=(0,))

    context_features_per_image[i, : context_lengths[i], :] = _context_features
    context_attention_mask_per_image[i, : context_lengths[i]] = 1
```

**What:** Extract text context **before each image** for image generation conditioning

**Example:**
```
Sequence: [<bos>, "A", "photo", "of", <|soi|>, <|image|>×77, "cat"]
                                       ↑
                            Image to generate

Context for image: ["<|soi|>", "of", "photo", "A", "<bos>"]  (reversed!)
```

**Why reverse?**
- Perceiver Resampler uses **learnable queries**
- Most recent tokens (closest to `<|soi|>`) should be attended to more
- Reversing puts them at the start

```python
# Add sincos position embeddings
pos_embed_1d = get_1d_sincos_pos_embed_from_grid(C, np.arange(seq_len))
context_features_per_image = self.context_feat_proj(context_features_per_image)
context_features_per_image += pos_embed_1d[None, :L_max]
```

**Output:** `[N_images, L_max, C]` context features with position embeddings

---

### **Step 6.2: Prepare MMFS Features for Image Decoder**

**Location:** Lines 560-569

```python
multiscale_features = output.pop("multiscale_features")  # From VisualTokenizer
(
    mmfs_features,
    mmfs_mask,
) = self._prepare_mmfs_features_for_image_decoder(
    multiscale_features,
    text_ids=text_ids,
    nearest_bos_idxs=nearest_bos_idxs,
    num_image_per_seq=num_image_per_seq,
)
```

### 🔍 Deep Dive: `_prepare_mmfs_features_for_image_decoder()`

**Location:** `uni_interleaved.py:352-386`

```python
# Build mask: which images can condition each target image?
image_context_mask = nearest_bos_idxs[:, None] <= image_start_token_idx[None, :]
image_context_mask = torch.tril(image_context_mask, diagonal=-1)  # Lower triangular
image_context_mask = torch.triu(image_context_mask, diagonal=-1)  # Only diagonal=-1

# For each target image, gather features from preceding image
for i in range(N_images):
    image_context_idxs = image_context_mask[i].nonzero(as_tuple=True)[-1]
    for ms_feat, mmfs_feat in zip(multiscale_features, mmfs_features):
        mmfs_feat[i, : len(image_context_idxs)] = ms_feat[image_context_idxs]
    mmfs_mask[i, : len(image_context_idxs)] = 1
```

**What:** For interleaved image generation, use **previous image features** as context

**Example:**
```
Sequence: <img0> "A cat" <img1> "chasing a dog" <img2>

For <img0>: No previous images → mmfs_features = zeros
For <img1>: Previous image = img0 → mmfs_features = features(img0)
For <img2>: Previous image = img1 → mmfs_features = features(img1)
```

**Why?**
- **Consistency:** Generated images should be consistent with previous images
- **Context:** "chasing a dog" image should reference the "cat" from img0

---

### **Step 6.3: Forward Through Image Decoder**

**Location:** Lines 571-587

```python
loss_img = self.image_decoder(
    decoder=self.visual_tokenizer.encoder,  # StableDiffusion instance
    image_tensors=image_tensors if image_tensors_dec is None else image_tensors_dec,
    context_features=context_features,            # [N_img, L_ctx, C]
    context_attention_mask=context_attention_mask,  # [N_img, L_ctx]
    image_loss_mask=image_loss_mask,              # Optional: focus loss on regions
    mmfs_features=mmfs_features,                  # [N_img, 1, ...] from prev images
    mmfs_mask=mmfs_mask,                          # [N_img, 1]
)  # Returns: diffusion loss
```

### 🔍 What Happens Inside `ImageDecoder.forward()`?

**Location:** `uni_interleaved/models/decoders/decoder_image.py`

```python
# 1. Encode image to latent space
latents = decoder.vae.encode(image_tensors).latent_dist.sample()
latents = latents * 0.18215  # Scaling factor

# 2. Sample random timesteps
timesteps = torch.randint(0, 1000, (B,), device=latents.device)

# 3. Add noise to latents
noise = torch.randn_like(latents)
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# 4. Compress context with Perceiver Resampler
encoder_hidden_states = self.perceiver_resampler(
    encoder_hidden_states=context_features,
    encoder_attention_mask=context_attention_mask
)  # [N_img, 77, 1024]

# 5. UNet forward with MMFS injection
noise_pred = decoder.unet(
    noisy_latents,
    timesteps,
    encoder_hidden_states,  # Text context (77 tokens)
    mmfs_features,          # Multi-scale image features
    mmfs_mask,              # Valid feature mask
)

# 6. Compute MSE loss
loss = F.mse_loss(noise_pred, noise)
```

**Process Flow:**
```
Ground Truth Image (512×512)
    ↓ [VAE Encoder]
Latent (64×64)
    ↓ [Add Noise]
Noisy Latent
    ↓ [UNet with context+MMFS]
Predicted Noise
    ↓ [MSE Loss]
Diffusion Loss
```

**MMFS Injection in UNet:**
```python
# At each UNet block:
unet_hidden = unet_layer(unet_hidden)

# Inject MMFS features at matching spatial scales
if unet_hidden.shape[-1] in [64, 32, 16, 8]:
    mmfs_injection = mmfs_net(
        unet_hidden,
        mmfs_features,  # Multi-scale features from visual encoder
        mmfs_mask
    )
    unet_hidden = unet_hidden + mmfs_injection
```

---

### 📍 Back to Main Forward: Lines 583-587

```python
# Weight and accumulate image loss
loss_img_weight = (
    loss_img_weight if loss_img_weight is not None else self.loss_img_weight
)  # Default: 5.0
loss = loss + loss_img.mean() * loss_img_weight

output["loss_img"] = loss_img.mean().detach()
```

---

## Step 7: Return Combined Loss

### 📍 Location: Lines 589-590

```python
output["loss"] = loss
return output
```

### 📤 Final Output Dictionary

```python
{
    "loss": total_loss,           # Weighted sum of all losses
    "loss_txt": text_loss,        # Text generation loss (×1.0)
    "loss_img": image_loss,       # Image generation loss (×5.0)
    "loss_sniffer": sniffer_loss  # Diffusion feedback loss (×5.0)
}
```

**Default Loss Weighting:**
```python
total_loss = (
    loss_txt * 1.0 +
    loss_img * 5.0 +
    loss_sniffer * 5.0
)
```

**Why these weights?**
- **Text loss (1.0):** Standard language modeling
- **Image loss (5.0):** Image generation is harder, needs more emphasis
- **Sniffer loss (5.0):** Visual understanding feedback is critical

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│ • text_ids: [B, L] with <|image|> placeholders            │
│ • images: [N_img, 3, 224, 224] (encoder)                  │
│ • images_dec: [N_img, 3, 512, 512] (decoder)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Prepare Multimodal Embeddings                      │
│ ├─ Encode images → 77 visual tokens per image             │
│ ├─ Replace <|image|> tokens with visual embeddings        │
│ ├─ Add learnable <|soi|> embeddings                       │
│ ├─ Prepare cross-attention masks                           │
│ └─ Prepare multi-scale features for MMFS                   │
│                                                             │
│ Output: mm_embeds [B, L, 4096]                            │
│         mmfs_features [B, N_img, 1344, 1024]              │
│         loss_sniffer (diffusion feedback)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: LLM Forward with Cross-Attention                   │
│ ├─ Self-attention on mm_embeds                             │
│ ├─ Cross-attention with mmfs_features (every 4th layer)   │
│ └─ Output: mm_hidden_state [B, L, 4096]                   │
└─────────────────────────────────────────────────────────────┘
                    ↓                    ↓
       ┌────────────────────┐  ┌────────────────────┐
       │ STEP 3: Text Path  │  │ STEP 4: Image Path │
       └────────────────────┘  └────────────────────┘
                 ↓                        ↓
    ┌───────────────────────┐  ┌──────────────────────────┐
    │ Text Decoder (LM Head)│  │ Image Decoder (SD 2.1)   │
    │ → Logits [B,L,32006] │  │ → Diffusion Loss         │
    └───────────────────────┘  └──────────────────────────┘
                 ↓                        ↓
    ┌───────────────────────┐  ┌──────────────────────────┐
    │ Cross-Entropy Loss    │  │ MSE Noise Prediction     │
    │ loss_txt × 1.0        │  │ loss_img × 5.0           │
    └───────────────────────┘  └──────────────────────────┘
                    ↓                     ↓
                    └─────────┬───────────┘
                              ↓
            ┌──────────────────────────────────┐
            │ STEP 5: Combine Losses           │
            │ total_loss =                     │
            │   loss_txt × 1.0 +               │
            │   loss_img × 5.0 +               │
            │   loss_sniffer × 5.0             │
            └──────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ OUTPUT          │
                    │ • loss: scalar  │
                    │ • loss_txt      │
                    │ • loss_img      │
                    │ • loss_sniffer  │
                    └─────────────────┘
```

---

## 🎓 Key Takeaways

1. **Multimodal Embedding Fusion:**
   - `<|image|>` tokens serve as placeholders
   - Replaced with actual 77-token visual embeddings from VisualTokenizer
   - Enables seamless text-image interleaving

2. **Cross-Attention Architecture:**
   - LLM attends to multi-scale visual features every 4 layers
   - MMFS enables efficient multi-image, multi-scale attention
   - Causal masking ensures proper temporal ordering

3. **Three-Headed Loss:**
   - **Text loss:** Standard language modeling
   - **Image loss:** Diffusion-based image generation
   - **Sniffer loss:** Diffusion feedback for visual understanding

4. **Context Extraction:**
   - Text context before `<|soi|>` conditions image generation
   - Previous image features condition subsequent image generation
   - Enables coherent interleaved generation

5. **Training Stability:**
   - Careful loss masking (ignore prompt, padding, special tokens)
   - Gradient checkpointing for memory efficiency
   - Weighted losses balance text and image generation

---

## 📚 Related Documentation

- [Top 50 Critical APIs](../05_references/TOP_50_CRITICAL_APIS.md)
- [MMFS Mechanism Deep Dive](../04_mechanisms/MMFS_EXPLAINED.md)
- [Visual Tokenizer Tutorial](../03_tutorials/01_visual_tokenizer.md)
- [Image Decoder Tutorial](../03_tutorials/05_image_generation.md)

---

**Last Updated:** 2025-11-24
**Author:** Claude Code
**Lines of Code Explained:** 590 (100% of forward pass)
