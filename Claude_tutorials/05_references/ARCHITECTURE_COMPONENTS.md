# DEEM Architecture Components Reference

## Overview
This document maps the **5 core architectural components** of the DEEM multimodal VLM to their file paths and primary roles.

---

## 📋 Core Component Mapping Table

| Component | File Path | Primary Role | Input → Output |
|-----------|-----------|--------------|----------------|
| **1. Vision Encoder (Sniffer)** | `uni_interleaved/models/encoders/visual_tokenizer.py` | Encodes raw images into visual embeddings with diffusion feedback | Image Tensor (3×224×224) → Visual Embeddings (B×257×1024) |
| **2. Connector/Projection Layer (Perceiver Resampler)** | `uni_interleaved/models/decoders/perceiver.py` | Compresses variable-length visual features to fixed tokens | Visual Features (B×257×1024) → Compressed Tokens (B×77×1024) |
| **3. LLM Backbone (LLaMA with Cross-Attention)** | `uni_interleaved/models/decoders/modeling_llama_mmfs.py` | Processes interleaved text+vision tokens with cross-attention | MM Embeddings → Contextualized Features |
| **4. Text Decoder** | `uni_interleaved/models/decoders/decoder_text.py` | Generates text tokens from LLM hidden states | Hidden States → Text Token Logits |
| **5. Image Decoder (Stable Diffusion)** | `uni_interleaved/models/decoders/sd.py` | Generates images conditioned on LLM context | Context Features + Noise → Generated Images |

---

## 🔍 Detailed Component Breakdown

### 1. Vision Encoder (VisualTokenizer)

**Location:** `uni_interleaved/models/encoders/visual_tokenizer.py:96-446`

**Key Classes:**
- `VisualTokenizer` - Main vision encoding module

**Sub-Components:**
- **Sniffer (CLIP Encoder):**
  - ViT Adapter: `uni_interleaved/models/encoders/vit_adapter/vit_adapter_hf.py`
  - ConvNeXT Adapter: `uni_interleaved/models/encoders/convnext_adapter/convnext_adapter_timm.py`
- **Diffusion Encoder (SD 2.1):** `uni_interleaved/models/decoders/sd.py`
- **Feature Extractor:** `MaskPooling` class for region-based visual understanding

**Key Methods:**
- `forward()` - Main encoding with optional diffusion feedback (line 315)
- `tta()` - Test-time adaptation using diffusion guidance (line 204)

**Configuration:**
```python
{
    "sniffer_model_path": "./assets/openai/clip-vit-large-patch14",
    "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
    "image_size": 512,
    "clip_normalize": True,
    "grid_size": 16  # 16×16 spatial grid
}
```

---

### 2. Connector Layer (Perceiver Resampler)

**Location:** `uni_interleaved/models/decoders/perceiver.py:7-31`

**Key Classes:**
- `PerceiverResampler` - BLIP-2 QFormer-based resampler

**Architecture:**
- Input: Variable-length visual features (e.g., 257 tokens from ViT)
- Output: **77 fixed tokens** (matching Stable Diffusion's text encoder output)
- Uses learnable queries + cross-attention to compress features

**Key Methods:**
- `forward()` - Cross-attention compression (line 27)

**Configuration:**
```python
{
    "num_queries": 77,  # Output token count
    "hidden_size": 1024,
    "qk_normalization": True,
    "gradient_checkpointing": True
}
```

**Role in Pipeline:**
```
CLIP Features (257 tokens) → Perceiver Resampler → 77 Image Tokens → LLM
```

---

### 3. LLM Backbone (LLaMA with MMFS Cross-Attention)

**Location:** `uni_interleaved/models/decoders/modeling_llama_mmfs.py`

**Key Classes:**
- `LlamaForCausalLM` - Modified LLaMA with cross-attention layers
- `LlamaModel` - Base model with `llama_cross_attn` modules

**Special Modifications:**
- **Cross-Attention Frequency:** Every 4th layer injects visual features
- **MMFS Integration:** Multi-scale multi-image feature synchronization
- **Flash Attention:** Memory-efficient attention via monkey patches

**Key Configuration:**
```python
llm_config.image_embed_dim = 1024
llm_config.cross_attention_frequency = 4  # Cross-attn every 4 layers
llm_config.spatial_shapes = [32, 16, 8]  # Multi-scale feature levels
```

**Visual Integration:**
```python
# In each cross-attention layer:
hidden_states = self.llama_cross_attn(
    hidden_states,           # Text embeddings
    vision_hidden_states,    # Multi-scale image features
    cross_attention_mask     # Attention mask
)
```

---

### 4. Text Decoder

**Location:** `uni_interleaved/models/decoders/decoder_text.py`

**Key Classes:**
- `TextDecoder` - LM head for text generation

**Vocabulary:**
- Base: 32,000 tokens (Vicuna tokenizer)
- Extended: **32,006 tokens** with special tokens:
  - `<|soi|>` (32000) - Start of image
  - `<|image|>` (32001) - Image placeholder (×77 per image)
  - `<refleft>`, `<refright>` - Reference box delimiters
  - `<boxleft>`, `<boxright>` - Bounding box coordinates

**Key Methods:**
- `init_from_llm()` - Initialize from LLaMA weights
- `forward()` - Compute token logits

---

### 5. Image Decoder (Stable Diffusion 2.1)

**Location:** `uni_interleaved/models/decoders/sd.py`

**Key Classes:**
- `StableDiffusion` - Full SD 2.1 integration
- `MMFSNet` - Multi-scale feature fusion module

**Components:**
- **VAE:** Encoder/Decoder for latent space (8× compression)
- **UNet:** Noise prediction network with MMFS injection
- **Scheduler:** DDPM for denoising process

**Key Methods:**
- `forward()` - Training with diffusion loss (line 390)
- `generate_images()` - Inference sampling (line 670)

**MMFS Integration:**
```python
# Multi-scale features from visual encoder injected at multiple UNet layers
unet_output = self.unet(
    latents,
    timesteps,
    encoder_hidden_states,  # From Perceiver (77 tokens)
    mmfs_features,          # From visual encoder (multi-scale)
    mmfs_mask
)
```

---

## 🔗 Data Flow Through Components

### Training Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INPUT: Image (3×512×512) + Text Tokens                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. VISION ENCODER (VisualTokenizer)                             │
│    • CLIP ViT/ConvNeXT: Image → Features (257×1024)            │
│    • Diffusion Feedback: SD Encoder → Sniffer Loss             │
│    • MaskPooling: Region-aware feature extraction              │
│    • Multi-scale features: [32×32, 16×16, 8×8] grids           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CONNECTOR (Perceiver Resampler)                              │
│    • Input: 257 visual tokens                                   │
│    • Output: 77 compressed tokens (for LLM + SD conditioning)   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. LLM BACKBONE (LLaMA + MMFS Cross-Attention)                  │
│    • Text Embeddings + Visual Tokens (77 per image)            │
│    • Cross-attention every 4 layers with multi-scale features   │
│    • Output: Contextualized hidden states                       │
└─────────────────────────────────────────────────────────────────┘
                    ↓                        ↓
        ┌───────────────────┐    ┌──────────────────────┐
        │ TEXT DECODER      │    │ IMAGE DECODER (SD)   │
        │ Hidden → Logits   │    │ Context → Image      │
        │ Loss: Cross-Ent   │    │ Loss: Diffusion      │
        └───────────────────┘    └──────────────────────┘
```

### Inference Pipeline

```
User Input: "Describe this image: [image.jpg]"
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. Image Preprocessing                                          │
│    • Dual-resolution: 224×224 (encoder), 512×512 (decoder)     │
│    • CLIP normalization for sniffer                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Visual Tokenization                                          │
│    • VisualTokenizer: Image → 77 tokens                        │
│    • Multi-scale features cached for later use                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Text Generation                                              │
│    • Prompt: "<bos> <|soi|> <|image|>×77 Describe this image:"│
│    • LLM generates: "A cat sitting on a mat. <|soi|>"          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Image Generation (if <|soi|> detected)                      │
│    • Extract context before <|soi|> token                      │
│    • Image Decoder: Context + Multi-scale features → Image     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Special Architectural Features

### Multi-Image Multi-Scale Feature Synchronizer (MMFS)

**Location:** `uni_interleaved/models/utils/ops/modules/mmfs.py:26-277`

**Purpose:** Handles multiple interleaved images in a single sequence with multi-scale attention

**Key Parameters:**
- `d_model=1024` - Feature dimension
- `n_levels=3` - Spatial scales [32, 16, 8]
- `n_heads=16` - Attention heads
- `n_points=8` - Sampling points per level
- `max_num_image_per_seq=50` - Max images per sequence

**Mechanism:**
```python
# Deformable attention over multiple images at multiple scales
output = MMFS(
    query=text_embeddings,           # [B, L, 1024]
    input_flatten=multi_scale_feats, # [B, N_images, HW, 1024]
    attention_mask=image_mask        # [B, L, N_images]
)
```

---

### Diffusion Feedback Mechanism

**Purpose:** Uses Stable Diffusion's encoding capability to provide feedback for visual understanding

**Training Loss:**
```python
# Visual tokenizer forward pass
visual_output = visual_tokenizer(image_enc, image_dec, mask)

# Diffusion feedback loss (weighted by mask)
sniffer_loss = SD_encoder_loss * mask_region * pos_weight
total_loss += sniffer_loss * 5.0  # loss_sniffer_weight
```

**Effect:** Forces the visual encoder to learn features that align with diffusion model's understanding

---

## 📊 Component Parameter Counts

| Component | Total Parameters | Trainable (Stage 2) |
|-----------|------------------|---------------------|
| Visual Tokenizer (Sniffer) | ~430M | 430M |
| Perceiver Resampler | ~150M | 150M |
| LLM Backbone (LLaMA-7B) | ~7B | ~200M (cross-attn only) |
| Text Decoder | ~7B | 7B (tied with LLM) |
| Image Decoder (SD 2.1) | ~1.2B | ~100M (MMFS modules) |
| **Total** | **~15.78B** | **~880M** |

---

## 🔧 Configuration Files

### Pretrain Config
`uni_interleaved/configs/train/pretrain.yaml`
- Stage 1: Image-text alignment
- Datasets: MMC4 + LAION
- Batch size: 128 global (4×32 GPUs)

### SFT VQA Config
`uni_interleaved/configs/train/sft_vqa.yaml`
- Stage 2: Visual question answering fine-tuning
- Datasets: VQAv2, TextVQA, OK-VQA, etc.

### Evaluation Config
`uni_interleaved/configs/eval/vqa_eval.yaml`
- VQA benchmarks
- Caption benchmarks
- Grounding benchmarks

---

## 🚀 Usage Example

```python
from uni_interleaved.models import MMInterleaved

# Initialize model
model = MMInterleaved(
    llm_model_path="./assets/lmsys/vicuna-7b-v1.5",
    seq_len=2048,
    num_img_token=77,
    image_embed_dim=1024,
    cross_attention_frequency=4,
    spatial_shapes=[32, 16, 8],
    visual_tokenizer_config={...},
    image_decoder_config={...}
)

# Forward pass
output = model(
    text_ids=text_ids,              # [B, L]
    image_tensors=images,           # [N_images, 3, 224, 224]
    image_tensors_dec=images_dec,   # [N_images, 3, 512, 512]
    num_image_per_seq=torch.tensor([2]),  # 2 images in sequence
    attention_mask=attn_mask
)

# Losses
loss_txt = output["loss_txt"]      # Text generation loss
loss_img = output["loss_img"]      # Image generation loss
loss_sniffer = output["loss_sniffer"]  # Diffusion feedback loss
```

---

## 📚 Related Documentation

- [Annotated Forward Pass](../02_annotated_code/ANNOTATED_FORWARD_PASS.md)
- [Visual Tokenizer Tutorial](../03_tutorials/01_visual_tokenizer.md)
- [MMFS Mechanism](../04_mechanisms/MMFS_EXPLAINED.md)

---

**Last Updated:** 2025-11-24
**Author:** Claude Code
**Repository:** DEEM (Diffusion models as Eyes of LLMs)
