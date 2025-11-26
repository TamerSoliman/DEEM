# DEEM Multimodal VLM Tutorial Collection

**Welcome to the comprehensive educational materials for DEEM** (Diffusion models serve as the Eyes of large language Models)!

This repository contains detailed code annotations, tutorials, and reference materials to help you understand and work with the DEEM multimodal architecture.

---

## 📚 What You'll Find Here

This tutorial collection provides:

1. **Architecture References** - Component mappings and API catalogs
2. **Annotated Code** - Line-by-line explanations of key functions
3. **Step-by-Step Tutorials** - Practical guides for common tasks
4. **Mechanism Deep Dives** - Detailed algorithm explanations
5. **Quick Reference Materials** - Tables, diagrams, and cheat sheets

---

## 🎯 Who This Is For

- **Researchers** studying multimodal VLMs
- **Engineers** implementing or fine-tuning DEEM
- **Students** learning about vision-language models
- **Practitioners** adapting DEEM for custom applications

---

## 🗺️ Tutorial Navigation

### 🏛️ **1. Architecture & References** (`05_references/`)

Start here to understand the big picture!

| Document | Description | Best For |
|----------|-------------|----------|
| [**Architecture Components**](05_references/ARCHITECTURE_COMPONENTS.md) | Maps 5 core components to file paths with data flow diagrams | Understanding overall structure |
| [**Top 50 Critical APIs**](05_references/TOP_50_CRITICAL_APIS.md) | Comprehensive catalog of user-facing classes and functions | API reference and navigation |

**Recommended First Read:** Architecture Components → Top 50 APIs

---

### 📖 **2. Annotated Code** (`02_annotated_code/`)

Deep dives into implementation with line-by-line explanations!

| Document | Lines Covered | Complexity | Time to Read |
|----------|---------------|------------|--------------|
| [**Annotated Forward Pass**](02_annotated_code/ANNOTATED_FORWARD_PASS.md) | 590 lines | ⭐⭐⭐⭐⭐ | 45 min |
| [**Annotated Visual Tokenizer**](02_annotated_code/ANNOTATED_VISUAL_TOKENIZER.md) | 446 lines | ⭐⭐⭐⭐ | 30 min |
| [**Annotated MMFS Module**](02_annotated_code/ANNOTATED_MMFS.md) | 277 lines | ⭐⭐⭐⭐⭐ | 40 min |
| [**Annotated Image Decoder**](02_annotated_code/ANNOTATED_IMAGE_DECODER.md) | 350 lines | ⭐⭐⭐⭐ | 35 min |
| [**Annotated Data Pipeline**](02_annotated_code/ANNOTATED_DATA_PIPELINE.md) | 400 lines | ⭐⭐⭐ | 25 min |

**Recommended Path:**
1. **Forward Pass** (understand overall flow)
2. **Visual Tokenizer** (visual encoding)
3. **MMFS Module** (cross-attention mechanism)
4. **Image Decoder** (generation)

---

### 🎓 **3. Step-by-Step Tutorials** (`03_tutorials/`)

Practical guides for implementing and using DEEM!

#### **Getting Started (Beginner)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**01 - Environment Setup**](03_tutorials/01_environment_setup.md) | Installation, dependencies, model downloads | 15 min |
| [**02 - Quick Start Inference**](03_tutorials/02_quick_start_inference.md) | Run your first inference in 5 minutes | 10 min |
| [**03 - Understanding Special Tokens**](03_tutorials/03_special_tokens.md) | `<|soi|>`, `<|image|>`, and token mechanics | 15 min |

#### **Core Components (Intermediate)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**04 - Visual Tokenizer Deep Dive**](03_tutorials/04_visual_tokenizer.md) | How images become 77 tokens | 30 min |
| [**05 - Perceiver Resampler**](03_tutorials/05_perceiver_resampler.md) | Feature compression mechanism | 25 min |
| [**06 - MMFS Cross-Attention**](03_tutorials/06_mmfs_cross_attention.md) | Multi-image multi-scale attention | 40 min |
| [**07 - Text Generation Pipeline**](03_tutorials/07_text_generation.md) | VQA and captioning implementation | 30 min |
| [**08 - Image Generation Pipeline**](03_tutorials/08_image_generation.md) | Stable Diffusion integration | 35 min |

#### **Training & Fine-Tuning (Advanced)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**09 - Pretraining Guide**](03_tutorials/09_pretraining.md) | Stage 1: Image-text alignment on MMC4+LAION | 45 min |
| [**10 - VQA Fine-Tuning**](03_tutorials/10_vqa_finetuning.md) | Stage 2: Fine-tuning on VQA datasets | 40 min |
| [**11 - LoRA Efficient Fine-Tuning**](03_tutorials/11_lora_finetuning.md) | Parameter-efficient training | 35 min |
| [**12 - Custom Dataset Integration**](03_tutorials/12_custom_datasets.md) | Add your own data | 30 min |
| [**13 - Loss Function Breakdown**](03_tutorials/13_loss_functions.md) | Understanding the 3-headed loss | 25 min |

#### **Data & Preprocessing (Intermediate)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**14 - Data Collators Explained**](03_tutorials/14_data_collators.md) | VQA, Caption, Grounding collators | 30 min |
| [**15 - Dual-Resolution Transform**](03_tutorials/15_dual_resolution.md) | 224×224 encoder + 512×512 decoder | 20 min |
| [**16 - WebDataset Streaming**](03_tutorials/16_webdataset_streaming.md) | Efficient large-scale data loading | 25 min |

#### **Advanced Topics (Expert)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**17 - Diffusion Feedback**](03_tutorials/17_diffusion_feedback.md) | How SD provides visual understanding | 40 min |
| [**18 - Test-Time Adaptation**](03_tutorials/18_tta.md) | Gradient-based TTA with diffusion | 35 min |
| [**19 - Flash Attention Integration**](03_tutorials/19_flash_attention.md) | Memory-efficient attention | 25 min |
| [**20 - Multi-GPU Training**](03_tutorials/20_multi_gpu_training.md) | DDP and DeepSpeed setup | 30 min |

#### **Evaluation & Benchmarks (Intermediate)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**21 - VQA Evaluation**](03_tutorials/21_vqa_evaluation.md) | VQAv2, TextVQA, GQA metrics | 25 min |
| [**22 - Caption Metrics**](03_tutorials/22_caption_metrics.md) | BLEU, CIDEr, METEOR scoring | 20 min |
| [**23 - Grounding Evaluation**](03_tutorials/23_grounding_evaluation.md) | RefCOCO IoU metrics | 25 min |
| [**24 - Image Quality Metrics**](03_tutorials/24_image_quality.md) | FID, CLIP score for generation | 20 min |

#### **Practical Applications (All Levels)**

| Tutorial | Topic | Duration |
|----------|-------|----------|
| [**25 - Interleaved Generation**](03_tutorials/25_interleaved_generation.md) | Text → Image → Text chains | 30 min |
| [**26 - Referring Segmentation**](03_tutorials/26_referring_segmentation.md) | Mask-based visual grounding | 35 min |
| [**27 - Visual Question Answering**](03_tutorials/27_vqa_application.md) | End-to-end VQA implementation | 25 min |
| [**28 - Image Editing**](03_tutorials/28_image_editing.md) | Context-conditioned editing | 30 min |
| [**29 - Zero-Shot Transfer**](03_tutorials/29_zero_shot.md) | Apply to new tasks without training | 25 min |
| [**30 - Model Optimization**](03_tutorials/30_optimization.md) | Quantization, distillation, pruning | 40 min |

---

### 🔬 **4. Mechanism Deep Dives** (`04_mechanisms/`)

Understand the "why" and "how" behind key algorithms!

| Mechanism | Description | Difficulty |
|-----------|-------------|------------|
| [**MMFS Explained**](04_mechanisms/MMFS_EXPLAINED.md) | Multi-Image Multi-Scale Feature Synchronizer | ⭐⭐⭐⭐⭐ |
| [**Diffusion Feedback**](04_mechanisms/DIFFUSION_FEEDBACK.md) | How SD encoder provides visual understanding | ⭐⭐⭐⭐ |
| [**Deformable Attention**](04_mechanisms/DEFORMABLE_ATTENTION.md) | Learnable sampling offsets in MMFS | ⭐⭐⭐⭐⭐ |
| [**Perceiver Resampling**](04_mechanisms/PERCEIVER_RESAMPLING.md) | Cross-attention compression to 77 tokens | ⭐⭐⭐ |
| [**Multi-Scale Features**](04_mechanisms/MULTISCALE_FEATURES.md) | Why 32×32, 16×16, 8×8 grids | ⭐⭐⭐ |

---

## 🚀 Quick Start Paths

### Path 1: **I want to use DEEM for inference**
```
1. 01 - Environment Setup
2. 02 - Quick Start Inference
3. 07 - Text Generation Pipeline
4. 08 - Image Generation Pipeline
5. 25 - Interleaved Generation
```

### Path 2: **I want to understand the architecture**
```
1. Architecture Components (reference)
2. Annotated Forward Pass (code)
3. 04 - Visual Tokenizer Deep Dive
4. 06 - MMFS Cross-Attention
5. MMFS Explained (mechanism)
```

### Path 3: **I want to fine-tune on my data**
```
1. 10 - VQA Fine-Tuning
2. 11 - LoRA Efficient Fine-Tuning
3. 12 - Custom Dataset Integration
4. 13 - Loss Function Breakdown
5. 20 - Multi-GPU Training
```

### Path 4: **I want to contribute or modify**
```
1. Architecture Components
2. Top 50 Critical APIs
3. Annotated Forward Pass
4. All Mechanism Deep Dives
5. 30 - Model Optimization
```

---

## 📊 Tutorial Metrics

| Category | # Tutorials | Total Time | Avg Difficulty |
|----------|-------------|------------|----------------|
| Getting Started | 3 | 40 min | ⭐⭐ |
| Core Components | 5 | 2h 40min | ⭐⭐⭐⭐ |
| Training | 5 | 3h 15min | ⭐⭐⭐⭐⭐ |
| Data | 3 | 1h 15min | ⭐⭐⭐ |
| Advanced | 4 | 2h 10min | ⭐⭐⭐⭐⭐ |
| Evaluation | 4 | 1h 30min | ⭐⭐⭐ |
| Applications | 6 | 3h | ⭐⭐⭐ |
| **Total** | **30** | **~14h** | **⭐⭐⭐⭐** |

---

## 🛠️ Practical Resources

### Code Snippets

Each tutorial includes:
- ✅ **Copy-paste code examples**
- ✅ **Jupyter notebook versions** (where applicable)
- ✅ **Common errors and fixes**
- ✅ **Performance tips**

### Visualizations

- Architecture diagrams
- Data flow charts
- Attention pattern visualizations
- Loss curve examples

### Cheat Sheets

- [Special Token Reference](05_references/SPECIAL_TOKENS_CHEATSHEET.md)
- [Configuration Parameters](05_references/CONFIG_PARAMS_CHEATSHEET.md)
- [Common Commands](05_references/COMMANDS_CHEATSHEET.md)

---

## 📖 DEEM Paper Summary

**Title:** "DEEM: Diffusion Models Serve as the Eyes of Large Language Models for Image Perception"

**Key Contributions:**

1. **Diffusion Feedback:** Uses Stable Diffusion as a visual encoder to provide feedback for CLIP-based features
2. **MMFS Module:** Multi-Image Multi-Scale Feature Synchronizer for handling multiple interleaved images
3. **Unified Generation:** Single model for text, image, and score generation

**Three-Stage Training:**
1. **Pretraining:** Image-text alignment on MMC4 + LAION
2. **Image-Text SFT:** VQA, captioning on LLaVA-665k + VQA datasets
3. **Mask-Text SFT:** Referring segmentation on RefCOCO + VG

**Architecture Highlights:**
- **Vision Encoder:** CLIP ViT/ConvNeXT + Diffusion feedback
- **LLM Backbone:** Vicuna-7B (LLaMA-based) with cross-attention every 4 layers
- **Image Decoder:** Stable Diffusion 2.1 with MMFS injection

---

## 🎓 Learning Outcomes

After completing these tutorials, you will be able to:

✅ **Understand** the full multimodal pipeline from images to text/image generation
✅ **Implement** custom datasets and collators for your data
✅ **Fine-tune** DEEM on domain-specific tasks with LoRA or full fine-tuning
✅ **Optimize** training and inference for your hardware
✅ **Debug** common issues with data, training, and generation
✅ **Extend** DEEM with custom components or modifications
✅ **Evaluate** model performance on standard benchmarks
✅ **Deploy** DEEM for production applications

---

## 🤝 How to Use These Tutorials

### For Beginners
1. Start with **Architecture Components** for a high-level overview
2. Read **02 - Quick Start Inference** to get hands-on experience
3. Work through **Getting Started** tutorials sequentially
4. Refer to **Annotated Forward Pass** when you want deeper understanding

### For Intermediate Users
1. Skim **Top 50 Critical APIs** to understand codebase structure
2. Jump to specific tutorials based on your needs (see Quick Start Paths)
3. Use **Annotated Code** as reference when reading source files
4. Consult **Mechanism Deep Dives** for algorithm details

### For Advanced Users
1. Read **all Annotated Code** documents for complete understanding
2. Study **Mechanism Deep Dives** to master core algorithms
3. Use tutorials as templates for custom implementations
4. Contribute back: improve tutorials or add new ones!

---

## 📝 Tutorial Conventions

### Code Formatting

```python
# Actual code from repository
def forward(self, x):
    return self.layer(x)
```

```python
# Simplified pseudocode for explanation
function process_image(image):
    features = encode(image)
    tokens = compress(features)
    return tokens
```

### Annotations

- **What:** Description of what the code does
- **Why:** Rationale for design decisions
- **How:** Implementation details and mechanics
- **When:** Conditions under which code executes
- **Where:** File path and line numbers
- **Context:** How this fits into larger pipeline

### Difficulty Levels

- ⭐ **Beginner:** Basic Python, minimal ML background
- ⭐⭐ **Novice:** Familiar with PyTorch, basic transformers
- ⭐⭐⭐ **Intermediate:** Understand attention, LLMs, vision models
- ⭐⭐⭐⭐ **Advanced:** Deep knowledge of transformers, diffusion models
- ⭐⭐⭐⭐⭐ **Expert:** Research-level understanding of multimodal VLMs

---

## 🔧 Prerequisites

### Software

- Python 3.10+
- PyTorch 2.1.0+
- CUDA 12.1+ (for GPU training)
- 64GB+ RAM (for full model)
- 4× A100 80GB GPUs (recommended for training)

### Knowledge

**Minimum:**
- Python programming
- Basic PyTorch (tensors, modules, optimizers)
- Transformers basics (attention, embeddings)

**Recommended:**
- Vision transformers (ViT, CLIP)
- Large language models (LLaMA, Vicuna)
- Diffusion models (Stable Diffusion)
- Multi-GPU training (DDP)

---

## 📚 Additional Resources

### Papers

- **DEEM Paper:** [arXiv:2405.15232](https://arxiv.org/abs/2405.15232)
- **LLaVA:** Visual instruction tuning foundations
- **BLIP-2:** Perceiver resampler architecture
- **Stable Diffusion:** Diffusion model background
- **Deformable DETR:** Deformable attention mechanism

### Related Repositories

- [MM-Interleaved](https://github.com/OpenGVLab/MM-Interleaved) - DEEM is built upon this
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Visual instruction tuning
- [Osprey](https://github.com/CircleRadon/Osprey) - Referring segmentation
- [Stable Diffusion](https://github.com/huggingface/diffusers) - Diffusion models

### Community

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share use cases
- **Paper Authors:** Contact for research collaborations

---

## 🎯 Tutorial Roadmap

### Completed ✅
- Architecture references
- Annotated forward pass
- Core API catalog

### In Progress 🚧
- Mechanism deep dives
- Step-by-step tutorials
- Code snippets and examples

### Planned 📋
- Video walkthroughs
- Interactive Jupyter notebooks
- Case studies and applications
- Troubleshooting guide

---

## 🤝 Contributing

Want to improve these tutorials?

1. **Fix errors:** Submit issues for technical inaccuracies
2. **Add examples:** Contribute code snippets or case studies
3. **Improve clarity:** Suggest rewording or additional explanations
4. **New tutorials:** Propose topics not yet covered

---

## 📄 License

These tutorials are provided under the same license as the DEEM repository (Apache 2.0).

---

## 🙏 Acknowledgments

These tutorials were created to help the community understand and use DEEM. Special thanks to:

- DEEM paper authors for the innovative architecture
- MM-Interleaved team for the codebase foundation
- HuggingFace for transformers and diffusers libraries
- All open-source contributors

---

## 📬 Contact & Support

- **Documentation Issues:** Open GitHub issue with `docs` label
- **Tutorial Requests:** Open GitHub issue with `tutorial-request` label
- **General Questions:** Use GitHub Discussions

---

**Happy Learning! 🎓**

*Last Updated: 2025-11-24*
*Tutorial Collection Version: 1.0*
*DEEM Repository: [RainBowLuoCS/DEEM](https://github.com/RainBowLuoCS/DEEM)*
