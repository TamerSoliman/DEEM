# Top 50 Critical User-Facing APIs in DEEM

This document lists the **50 most critical classes and functions** that define the DEEM VLM workflow, organized by category.

---

## 📌 Quick Reference Table

| # | API Name | File Path | Category | Primary Purpose |
|---|----------|-----------|----------|-----------------|
| 1 | `MMInterleaved` | `uni_interleaved/models/uni_interleaved.py:25` | Core Model | Main multimodal model class |
| 2 | `MMInterleaved.forward()` | `uni_interleaved/models/uni_interleaved.py:454` | Core Model | Training forward pass with losses |
| 3 | `MMInterleaved.generate()` | `uni_interleaved/models/uni_interleaved.py:856` | Core Model | Unified generation interface |
| 4 | `MMInterleaved.generate_texts()` | `uni_interleaved/models/uni_interleaved.py:683` | Core Model | Text generation mode |
| 5 | `MMInterleaved.generate_images()` | `uni_interleaved/models/uni_interleaved.py:592` | Core Model | Image generation mode |
| 6 | `MMInterleaved.generate_scores()` | `uni_interleaved/models/uni_interleaved.py:759` | Core Model | Scoring/classification mode |
| 7 | `VisualTokenizer` | `uni_interleaved/models/encoders/visual_tokenizer.py:96` | Vision Encoder | Visual encoding with diffusion feedback |
| 8 | `VisualTokenizer.forward()` | `uni_interleaved/models/encoders/visual_tokenizer.py:315` | Vision Encoder | Encode images to visual tokens |
| 9 | `VisualTokenizer.tta()` | `uni_interleaved/models/encoders/visual_tokenizer.py:204` | Vision Encoder | Test-time adaptation with diffusion |
| 10 | `CLIPVisionTransformerAdapter` | `uni_interleaved/models/encoders/vit_adapter/vit_adapter_hf.py` | Vision Encoder | ViT-based multi-scale adapter |
| 11 | `CLIPVisionConvNextAdapter` | `uni_interleaved/models/encoders/convnext_adapter/convnext_adapter_timm.py` | Vision Encoder | ConvNeXT-based multi-scale adapter |
| 12 | `MaskPooling` | `uni_interleaved/models/encoders/visual_tokenizer.py:40` | Vision Encoder | Region-aware feature extraction |
| 13 | `PerceiverResampler` | `uni_interleaved/models/decoders/perceiver.py:7` | Connector | Visual feature compression (257→77 tokens) |
| 14 | `PerceiverResampler.forward()` | `uni_interleaved/models/decoders/perceiver.py:27` | Connector | Cross-attention compression |
| 15 | `LlamaForCausalLM` | `uni_interleaved/models/decoders/modeling_llama_mmfs.py` | LLM Backbone | Modified LLaMA with cross-attention |
| 16 | `LlamaModel` | `uni_interleaved/models/decoders/modeling_llama_mmfs.py` | LLM Backbone | Base LLaMA with MMFS integration |
| 17 | `MMFS` | `uni_interleaved/models/utils/ops/modules/mmfs.py:26` | LLM Backbone | Multi-Image Multi-Scale Feature Synchronizer |
| 18 | `MMFS.forward()` | `uni_interleaved/models/utils/ops/modules/mmfs.py:120` | LLM Backbone | Deformable cross-attention over images |
| 19 | `TextDecoder` | `uni_interleaved/models/decoders/decoder_text.py` | Text Generation | Text generation head (LM head) |
| 20 | `TextDecoder.init_from_llm()` | `uni_interleaved/models/decoders/decoder_text.py` | Text Generation | Initialize from LLaMA weights |
| 21 | `ImageDecoder` | `uni_interleaved/models/decoders/decoder_image.py` | Image Generation | SD-based image generation |
| 22 | `ImageDecoder.forward()` | `uni_interleaved/models/decoders/decoder_image.py` | Image Generation | Training with diffusion loss |
| 23 | `ImageDecoder.generate_images()` | `uni_interleaved/models/decoders/decoder_image.py` | Image Generation | Inference image sampling |
| 24 | `StableDiffusion` | `uni_interleaved/models/decoders/sd.py` | Image Generation | Full SD 2.1 implementation |
| 25 | `MMFSNet` | `uni_interleaved/models/decoders/sd_mmfs.py` | Image Generation | Multi-scale feature fusion for UNet |
| 26 | `CascadeLlamaForCausalLMWrapper` | `uni_interleaved/models/utils/causal_lm_cascade.py` | Generation Utils | Combines mm_decoder + text_decoder for generation |
| 27 | `build_train_dataset()` | `uni_interleaved/custom_datasets/utils/build.py:252` | Data Loading | Constructs training datasets |
| 28 | `build_eval_dataset()` | `uni_interleaved/custom_datasets/utils/build.py:553` | Data Loading | Constructs evaluation datasets |
| 29 | `create_transform()` | `uni_interleaved/custom_datasets/utils/build.py:95` | Data Preprocessing | Creates image transform pipelines |
| 30 | `dual_transform` | `uni_interleaved/custom_datasets/utils/build.py:127` | Data Preprocessing | Dual-resolution transform (encoder+decoder) |
| 31 | `VQACaptionTrainCollator` | `uni_interleaved/custom_datasets/train/vqa_datasets.py` | Data Collation | Collates VQA/caption training data |
| 32 | `ImageTextPairTrainCollator` | `uni_interleaved/custom_datasets/train/pairs_datasets.py` | Data Collation | Collates image-text pairs |
| 33 | `GroundingTrainCollator` | `uni_interleaved/custom_datasets/train/grounding_datasets.py` | Data Collation | Collates grounding data with boxes |
| 34 | `ReferringMaskTrainCollator` | `uni_interleaved/custom_datasets/train/referring_mask_datasets.py` | Data Collation | Collates referring segmentation data |
| 35 | `RandomMixWdsDataset` | `uni_interleaved/custom_datasets/train/mix_dataset.py` | Data Mixing | Mixes multiple datasets with sampling |
| 36 | `WdsDataset` | `uni_interleaved/custom_datasets/utils/wds_utils.py` | Data Loading | WebDataset wrapper for streaming |
| 37 | `init_tokenizer()` | `uni_interleaved/custom_datasets/utils/wds_utils.py` | Tokenization | Initializes LLaMA tokenizer with special tokens |
| 38 | `LMMTrainer` | `uni_interleaved/engine/lmm_trainer.py` | Training | Custom trainer extending HF Trainer |
| 39 | `LMMTrainer.compute_loss()` | `uni_interleaved/engine/lmm_trainer.py` | Training | Computes multimodal training loss |
| 40 | `LMMTrainer.evaluation_loop()` | `uni_interleaved/engine/lmm_trainer.py` | Evaluation | Evaluation across multiple tasks |
| 41 | `TrainingArguments` | `uni_interleaved/utils/parse_args.py` | Training Config | Training configuration dataclass |
| 42 | `ArgumentParser` | `uni_interleaved/utils/parse_args.py` | Training Config | YAML + CLI argument parsing |
| 43 | `load_model_weights()` | `uni_interleaved/utils/misc.py` | Model Loading | Load checkpoint with position interpolation |
| 44 | `init_distributed_mode()` | `uni_interleaved/utils/misc.py` | Training Setup | Initialize DDP training |
| 45 | `replace_llama_attn_with_flash_attn()` | `uni_interleaved/models/utils/monkey_patch/__init__.py` | Optimization | Enables Flash Attention for LLaMA |
| 46 | `MSDeformAttnFunction` | `uni_interleaved/models/utils/ops/functions/ms_deform_attn_func.py` | MMFS CUDA | Multi-scale deformable attention CUDA kernel |
| 47 | `get_abs_pos()` | `uni_interleaved/models/utils/pos_embed.py` | Position Encoding | Interpolates sincos position embeddings |
| 48 | `train.py:main()` | `train.py:40` | Entry Point | Training script entry point |
| 49 | `inference.py:main()` | `inference.py:288` | Entry Point | Inference script entry point |
| 50 | `evaluate.py:main()` | `evaluate.py` | Entry Point | Evaluation script entry point |

---

## 🎯 Category 1: Core Model APIs (6 APIs)

### 1. `MMInterleaved` Class
**File:** `uni_interleaved/models/uni_interleaved.py:25`

```python
class MMInterleaved(nn.Module):
    def __init__(
        self,
        llm_model_path="",
        seq_len=2048,
        txt_vocab_size=32006,
        loss_img_weight=5.0,
        loss_txt_weight=1.0,
        loss_sniffer_weight=5.0,
        special_token_dict=dict(...),
        visual_tokenizer_config=None,
        image_decoder_config=None,
        num_img_token=64,
        image_embed_dim=1024,
        cross_attention_frequency=4,
        spatial_shapes=[32, 16, 8],
        freeze_llm=True,
        freeze_vfm=False,
        freeze_dm=True,
    )
```

**Purpose:** Main model class integrating all components
**Inputs:** Configuration parameters
**Outputs:** Initialized model with all sub-components

---

### 2. `MMInterleaved.forward()`
**File:** `uni_interleaved/models/uni_interleaved.py:454-590`

```python
def forward(
    self,
    text_ids: torch.LongTensor,           # [B, L] text token IDs
    image_tensors: torch.FloatTensor,     # [N_img, 3, H, W] images for encoder
    image_tensors_dec: torch.FloatTensor, # [N_img, 3, H, W] images for decoder
    num_image_per_seq: torch.Tensor,      # [B] number of images per sequence
    attention_mask: torch.Tensor,         # [B, L] attention mask
    **kwargs
) -> dict
```

**Returns:**
```python
{
    "loss": total_loss,           # Combined loss
    "loss_txt": text_loss,        # Text generation loss
    "loss_img": image_loss,       # Image generation loss
    "loss_sniffer": sniffer_loss  # Diffusion feedback loss
}
```

**Key Operations:**
1. Prepare multimodal embeddings (`_prepare_mm_embeds`)
2. Forward through LLM with cross-attention
3. Text decoder forward
4. Image decoder forward
5. Compute weighted losses

---

### 3. `MMInterleaved.generate()`
**File:** `uni_interleaved/models/uni_interleaved.py:856-871`

```python
def generate(
    self,
    mode="generate_images",  # or "generate_texts", "generate_scores"
    **kwargs
)
```

**Purpose:** Unified generation interface
**Modes:**
- `"generate_images"` → Image generation
- `"generate_texts"` or `"generate_vqa"` → Text generation
- `"generate_scores"` → Classification scoring

---

### 4. `MMInterleaved.generate_texts()`
**File:** `uni_interleaved/models/uni_interleaved.py:683-757`

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
    **kwargs
) -> dict
```

**Returns:**
```python
{
    "text_ids": generated_token_ids  # [B, max_length]
}
```

**Key Features:**
- Uses `CascadeLlamaForCausalLMWrapper` for generation
- Supports beam search and nucleus sampling
- Auto-stops at `<eos>` or `<|soi|>` tokens

---

### 5. `MMInterleaved.generate_images()`
**File:** `uni_interleaved/models/uni_interleaved.py:592-681`

```python
def generate_images(
    self,
    text_ids,
    image_tensors,
    num_image_per_seq,
    attention_mask,
    target_image_idxs=None,  # Which images to generate
    **kwargs
) -> dict
```

**Returns:**
```python
{
    "image": generated_images  # [N_target, 3, H, W]
}
```

**Process:**
1. Extract context features before `<|soi|>` tokens
2. Prepare multi-scale features for MMFS
3. Call `ImageDecoder.generate_images()`

---

### 6. `MMInterleaved.generate_scores()`
**File:** `uni_interleaved/models/uni_interleaved.py:759-854`

```python
def generate_scores(
    self,
    text_ids: List[torch.LongTensor],       # Contexts per sample
    image_tensors,
    options_ids: List[torch.LongTensor],    # Answer options
    options_attn_masks: List[torch.LongTensor],
    **kwargs
) -> dict
```

**Purpose:** Classification/scoring for datasets like VisDial
**Returns:** Log-probability scores for each option

---

## 🎯 Category 2: Vision Encoding APIs (6 APIs)

### 7. `VisualTokenizer` Class
**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:96-446`

```python
class VisualTokenizer(nn.Module):
    def __init__(
        self,
        sniffer_model_path="./assets/openai/clip-vit-large-patch14",
        perceiver_config=None,
        llm_hidden_size=5120,
        diffusion_hidden_size=1024,
        sd_use_encoder=False,  # Enable diffusion feedback
        **kwargs
    )
```

**Components:**
- `sniffer`: CLIP ViT or ConvNeXT encoder
- `encoder`: Stable Diffusion encoder (optional)
- `perceiver_resampler`: Compresses to 77 tokens
- `feature_extractor`: MaskPooling for region understanding

---

### 8. `VisualTokenizer.forward()`
**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:315-444`

```python
def forward(
    self,
    image,           # [B, 3, 224, 224] for CLIP
    image_dec,       # [B, 3, 512, 512] for SD
    image_mask,      # [B, 1, H, W] region mask
) -> dict
```

**Returns:**
```python
{
    "vis_embed": visual_tokens,        # [B, 77, llm_hidden_size]
    "multiscale_features": [feat1, feat2, feat3],  # Multi-scale
    "loss_sniffer": diffusion_feedback_loss
}
```

**Process:**
1. Normalize image with CLIP stats
2. CLIP encoding → multi-scale features
3. Add sincos position embeddings
4. Region masking (MaskPooling)
5. Perceiver resampler compression
6. Optional: Diffusion feedback loss

---

### 9. `VisualTokenizer.tta()`
**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:204-280`

```python
def tta(
    self,
    image, image_dec, image_mask,
    samples_num=9,   # Number of timestep samples
    steps_num=10     # Optimization steps
)
```

**Purpose:** Test-time adaptation using gradient-based diffusion feedback
**Process:**
1. Sample random diffusion timesteps
2. Forward through visual encoder
3. Compute diffusion loss as feedback
4. Backprop to optimize visual features
5. Restore original state

---

### 10-11. CLIP Adapters

**ViT Adapter:** `uni_interleaved/models/encoders/vit_adapter/vit_adapter_hf.py`
```python
class CLIPVisionTransformerAdapter(nn.Module):
    # Multi-scale ViT feature extraction with deformable attention
```

**ConvNeXT Adapter:** `uni_interleaved/models/encoders/convnext_adapter/convnext_adapter_timm.py`
```python
class CLIPVisionConvNextAdapter(nn.Module):
    # Multi-scale ConvNeXT feature extraction
```

---

### 12. `MaskPooling`
**File:** `uni_interleaved/models/encoders/visual_tokenizer.py:40-94`

```python
class MaskPooling(nn.Module):
    def forward(
        self,
        multi_scale_feats: List[Tensor],  # Features at different scales
        mask: Tensor                       # Binary mask [B, 1, H, W]
    ) -> Tensor  # [B, num_scales+1, hidden_dim]
```

**Purpose:** Extracts features weighted by mask regions
**Use Case:** Referring segmentation, part-based understanding

---

## 🎯 Category 3: Connector & LLM APIs (6 APIs)

### 13. `PerceiverResampler`
**File:** `uni_interleaved/models/decoders/perceiver.py:7-31`

```python
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_queries=32,      # Output token count (77 for DEEM)
        hidden_size=768,     # Feature dimension
        qk_normalization=False,
        gradient_checkpointing=True,
        **kwargs
    )
```

**Based on:** BLIP-2 QFormer
**Purpose:** Compress variable-length visual features to fixed tokens

---

### 14. `PerceiverResampler.forward()`
**File:** `uni_interleaved/models/decoders/perceiver.py:27-30`

```python
def forward(
    self,
    encoder_hidden_states,      # [B, N, hidden_size] visual features
    encoder_attention_mask=None,
    query_embeds=None,          # Learnable queries (default: self.queries)
    **kwargs
)
```

**Returns:** `(output_features, ...)` compressed to `num_queries` tokens

---

### 15-16. LLaMA with Cross-Attention

**LlamaForCausalLM:** Modified LLaMA with cross-attention layers
```python
# Every 4th layer:
hidden_states = llama_cross_attn(
    hidden_states,           # Text features
    vision_hidden_states,    # Multi-scale image features from MMFS
    cross_attention_mask     # [B, L, N_images]
)
```

**Configuration:**
- `image_embed_dim=1024`
- `cross_attention_frequency=4`
- `spatial_shapes=[32, 16, 8]`

---

### 17. `MMFS` Class
**File:** `uni_interleaved/models/utils/ops/modules/mmfs.py:26-277`

```python
class MMFS(nn.Module):
    """Multi-Image Multi-Scale Feature Synchronizer"""
    def __init__(
        self,
        d_model=256,
        n_levels=4,              # Spatial scales
        n_heads=8,               # Attention heads
        n_points=8,              # Sampling points per level
        spatial_shapes=[16],
        max_num_image_per_seq=50,
    )
```

**Purpose:** Deformable cross-attention for multiple images at multiple scales

---

### 18. `MMFS.forward()`
**File:** `uni_interleaved/models/utils/ops/modules/mmfs.py:120-276`

```python
def forward(
    self,
    query,                      # [B, L_query, C] text embeddings
    reference_points,           # [B, L_query, n_levels, 2] spatial refs
    input_flatten,              # [B, N_images, HW, C] multi-scale features
    input_spatial_shapes,       # [(H, W)] for each level
    attention_mask,             # [B, L_query, N_images]
) -> Tensor  # [B, L_query, C]
```

**Key Mechanism:** Learnable sampling offsets + deformable attention

---

## 🎯 Category 4: Generation APIs (7 APIs)

### 19. `TextDecoder`
**File:** `uni_interleaved/models/decoders/decoder_text.py`

```python
class TextDecoder(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        txt_vocab_size=32006,    # Extended vocab
        orig_txt_vocab_size=32000,
    )
```

**Special Tokens:**
- 32000: `<|soi|>` - Start of image
- 32001: `<|image|>` - Image placeholder token
- 32002-32005: Reference/box tokens

---

### 20. `TextDecoder.init_from_llm()`
```python
def init_from_llm(self, llm_model, orig_txt_vocab_size):
    # Copy weights from LLaMA's LM head
    # Extend embedding matrix for new tokens
```

---

### 21. `ImageDecoder`
**File:** `uni_interleaved/models/decoders/decoder_image.py`

```python
class ImageDecoder(nn.Module):
    def __init__(
        self,
        uncond_prob=0.1,         # Classifier-free guidance
        seq_len=77,              # Context length
        embed_dim=1024,
        perceiver_config=None,
        decoder=None,            # StableDiffusion instance
    )
```

---

### 22. `ImageDecoder.forward()`
```python
def forward(
    self,
    decoder,                   # SD instance
    image_tensors,             # Ground truth images
    context_features,          # LLM context [B, L_context, C]
    context_attention_mask,
    mmfs_features,             # Multi-scale features
    mmfs_mask,
) -> Tensor  # Diffusion loss
```

---

### 23. `ImageDecoder.generate_images()`
```python
def generate_images(
    self,
    decoder,
    context_features,
    context_attention_mask,
    mmfs_features,
    mmfs_mask,
    num_inference_steps=50,
    guidance_scale=7.5,
    **kwargs
) -> dict  # {"image": [B, 3, H, W]}
```

---

### 24. `StableDiffusion`
**File:** `uni_interleaved/models/decoders/sd.py`

**Components:**
- VAE: `AutoencoderKL`
- UNet: `UNet2DConditionModel` + MMFS injection
- Scheduler: `DDPMScheduler`

**Methods:**
- `forward()`: Training with diffusion loss
- `generate_images()`: DDPM sampling
- `generate_inpaint_images()`: Inpainting

---

### 25. `MMFSNet`
**File:** `uni_interleaved/models/decoders/sd_mmfs.py`

```python
class MMFSNet(nn.Module):
    # Injects multi-scale features into UNet blocks
    def forward(self, unet_features, mmfs_features, mmfs_mask):
        # Fuses at multiple UNet resolutions
```

---

### 26. `CascadeLlamaForCausalLMWrapper`
**File:** `uni_interleaved/models/utils/causal_lm_cascade.py`

```python
class CascadeLlamaForCausalLMWrapper(nn.Module):
    def __init__(self, mm_decoder, text_decoder):
        # Combines LlamaModel + TextDecoder for generation API compatibility
```

**Purpose:** Enables use of HuggingFace `generate()` method

---

## 🎯 Category 5: Data Loading APIs (10 APIs)

### 27. `build_train_dataset()`
**File:** `uni_interleaved/custom_datasets/utils/build.py:252-301`

```python
def build_train_dataset(config):
    # Supports:
    # - Single dataset
    # - Mixed datasets (random_mix)
    # - SFT datasets (concat)
    return dataset  # with .collator and .tokenizer
```

---

### 28. `build_eval_dataset()`
**File:** `uni_interleaved/custom_datasets/utils/build.py:553-560`

```python
def build_eval_dataset(config):
    # Returns dict of datasets if list config
    return {
        "vqav2": vqa_dataset,
        "nocaps": caption_dataset,
        ...
    }
```

---

### 29. `create_transform()`
**File:** `uni_interleaved/custom_datasets/utils/build.py:95-125`

```python
def create_transform(
    aug_type="numpy",         # or "dual_numpy"
    resolution=224,           # For encoder
    resolution2=512,          # For decoder (if dual)
    resize=True,
    center_crop=True,
    random_flip=True,
):
    # Returns transform callable
```

---

### 30. `dual_transform`
**File:** `uni_interleaved/custom_datasets/utils/build.py:127-165`

```python
class dual_transform:
    def __call__(self, pil_image):
        arr1 = self.transform1(pil_image)  # 224×224
        arr2 = self.transform2(pil_image)  # 512×512
        return arr1, arr2
```

**Purpose:** Creates two resolutions for encoder and decoder

---

### 31-34. Data Collators

**VQA Collator:**
```python
class VQACaptionTrainCollator:
    def __call__(self, batch):
        # Collates VQA samples with prompts
        # Returns: text_ids, images, attention_mask, etc.
```

**Image-Text Pair Collator:**
```python
class ImageTextPairTrainCollator:
    # For caption datasets
```

**Grounding Collator:**
```python
class GroundingTrainCollator:
    # For RefCOCO with bounding boxes
```

**Referring Mask Collator:**
```python
class ReferringMaskTrainCollator:
    # For segmentation masks
```

---

### 35. `RandomMixWdsDataset`
**File:** `uni_interleaved/custom_datasets/train/mix_dataset.py`

```python
class RandomMixWdsDataset:
    def __init__(
        self,
        datasets: List[Dataset],
        probs=None,           # Sampling probabilities
        sampling_type="sum",  # or "longest"
    )
```

**Purpose:** Mixes multiple WebDatasets with probability sampling

---

### 36. `WdsDataset`
**File:** `uni_interleaved/custom_datasets/utils/wds_utils.py`

```python
class WdsDataset:
    # Wrapper for webdataset shards
    # Used for MMC4 and LAION pretraining
```

---

### 37. `init_tokenizer()`
**File:** `uni_interleaved/custom_datasets/utils/wds_utils.py`

```python
def init_tokenizer(tokenizer_path):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # Add special tokens: <|soi|>, <|image|>, etc.
    return tokenizer
```

---

## 🎯 Category 6: Training & Evaluation APIs (5 APIs)

### 38. `LMMTrainer`
**File:** `uni_interleaved/engine/lmm_trainer.py`

```python
class LMMTrainer(Trainer):
    # Extends HuggingFace Trainer for multimodal training
    def compute_loss(self, model, inputs):
        # Returns combined loss from model.forward()
```

**Features:**
- Multi-GPU DDP support
- DeepSpeed Zero-1
- Custom evaluation loop
- Gradient checkpointing

---

### 39. `LMMTrainer.compute_loss()`
```python
def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    loss = outputs["loss"]
    return (loss, outputs) if return_outputs else loss
```

---

### 40. `LMMTrainer.evaluation_loop()`
```python
def evaluation_loop(self, dataloader, description):
    # Iterates over multiple eval datasets
    # Computes metrics: VQA accuracy, CIDEr, etc.
```

---

### 41-42. Configuration APIs

**TrainingArguments:**
```python
class TrainingArguments(HfTrainingArguments):
    load_from_args: Optional[str] = None
    # Extends HF args with custom fields
```

**ArgumentParser:**
```python
class ArgumentParser:
    def parse_args_with_config_file_into_dataclasses():
        # Parses YAML config + CLI args
        return train_args, model_config
```

---

## 🎯 Category 7: Utilities & Optimization APIs (8 APIs)

### 43. `load_model_weights()`
**File:** `uni_interleaved/utils/misc.py`

```python
def load_model_weights(model, checkpoint_path):
    # Loads checkpoint
    # Handles position embedding interpolation
    # Supports partial loading (missing keys OK)
```

---

### 44. `init_distributed_mode()`
**File:** `uni_interleaved/utils/misc.py`

```python
def init_distributed_mode():
    # Initializes DDP from environment variables
    # Sets CUDA devices, world size, rank
```

---

### 45. `replace_llama_attn_with_flash_attn()`
**File:** `uni_interleaved/models/utils/monkey_patch/__init__.py`

```python
def replace_llama_attn_with_flash_attn():
    # Monkey-patches LLaMA attention with Flash Attention
    # 2-4× speedup, lower memory
```

**Other Monkey Patches:**
- `replace_blip2_attn_with_qknorm_attn()` - QK normalization
- `replace_stable_diffusion_unet_forward()` - SD hooks

---

### 46. `MSDeformAttnFunction`
**File:** `uni_interleaved/models/utils/ops/functions/ms_deform_attn_func.py`

**Purpose:** CUDA kernel for multi-scale deformable attention (used by MMFS)

---

### 47. `get_abs_pos()`
**File:** `uni_interleaved/models/utils/pos_embed.py`

```python
def get_abs_pos(pos_embed, spatial_size):
    # Interpolates sincos position embeddings
    # Handles dynamic image sizes
```

---

### 48-50. Entry Point Scripts

**train.py:**
```python
def main():
    model = MMInterleaved(**config.model)
    trainer = LMMTrainer(model, ...)
    trainer.train()
```

**inference.py:**
```python
def main():
    model = MMInterleaved(**config.model)
    model.eval()
    outputs = model.generate(mode="generate_texts", ...)
```

**evaluate.py:**
```python
def main():
    model = MMInterleaved(**config.model)
    # Run evaluation on benchmarks
```

---

## 📊 API Usage Frequency

| Category | # APIs | % of Total |
|----------|--------|------------|
| Core Model | 6 | 12% |
| Vision Encoding | 6 | 12% |
| Connector & LLM | 6 | 12% |
| Generation | 7 | 14% |
| Data Loading | 10 | 20% |
| Training & Eval | 5 | 10% |
| Utilities | 8 | 16% |
| Entry Points | 3 | 6% |

---

## 🔗 API Relationships

```
Entry Point (train.py)
    ↓
build_train_dataset()
    ↓
LMMTrainer
    ↓
MMInterleaved.forward()
    ├─→ VisualTokenizer.forward()
    │       ├─→ CLIPVisionAdapter
    │       ├─→ MaskPooling
    │       ├─→ PerceiverResampler
    │       └─→ StableDiffusion (feedback)
    ├─→ LlamaModel (with MMFS cross-attn)
    ├─→ TextDecoder
    └─→ ImageDecoder
            └─→ StableDiffusion.generate_images()
```

---

## 📚 Next Steps

- **Annotated Code:** See `02_annotated_code/` for detailed line-by-line explanations
- **Tutorials:** See `03_tutorials/` for step-by-step guides
- **Mechanisms:** See `04_mechanisms/` for deep dives into algorithms

---

**Last Updated:** 2025-11-24
**Author:** Claude Code
