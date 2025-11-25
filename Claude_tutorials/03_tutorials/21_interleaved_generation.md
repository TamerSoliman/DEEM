# Interleaved Generation

**Difficulty:** ⭐⭐⭐⭐ Advanced
**Time:** 35 minutes

---

## Overview

Generate sequences with multiple images and text, enabling multi-turn visual dialogue and image editing workflows.

---

## Single-Turn vs Multi-Turn

### Single-Turn (Basic)

```python
# One image, one response
image = load_image("cat.jpg")
question = "What color is the cat?"

answer = model.generate_texts(
    images=image,
    prompt=f"Question: {question} Answer:",
    max_new_tokens=10
)
# Output: "orange"
```

### Multi-Turn (Interleaved)

```python
# Multiple images, multi-turn dialogue
images = [load_image("cat1.jpg"), load_image("cat2.jpg")]
dialogue = [
    {"role": "user", "content": "What color is the first cat?"},
    {"role": "assistant", "content": "Orange."},
    {"role": "user", "content": "What about the second cat?"},
]

answer = model.generate_texts_interleaved(
    images=images,
    dialogue=dialogue,
    max_new_tokens=10
)
# Output: "Gray."
```

---

## Implementation

**File:** `uni_interleaved/models/uni_interleaved.py:550-620`

### Format Interleaved Sequence

```python
def format_interleaved_sequence(images, dialogue):
    """
    images: List[PIL.Image]
    dialogue: List[dict] with keys "role" and "content"
    """
    sequence = "<bos> "

    image_idx = 0
    for turn in dialogue:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            # Insert image before user text
            if image_idx < len(images):
                sequence += "<|soi|> " + "<|image|>" * 77 + " "
                image_idx += 1

            sequence += f"User: {content} "

        elif role == "assistant":
            sequence += f"Assistant: {content} "

    # Final prompt for next response
    sequence += "Assistant:"

    return sequence, image_idx

# Example
images = [cat1, cat2]
dialogue = [
    {"role": "user", "content": "What color is this cat?"},
    {"role": "assistant", "content": "Orange."},
    {"role": "user", "content": "And this one?"},
]

sequence, num_images = format_interleaved_sequence(images, dialogue)
# "<bos> <|soi|> <|image|>×77 User: What color is this cat? Assistant: Orange. <|soi|> <|image|>×77 User: And this one? Assistant:"
```

---

## Multi-Image Generation

```python
def generate_multi_image_response(model, images, dialogue):
    """Generate response considering multiple images"""

    # Format sequence
    sequence, num_images_used = format_interleaved_sequence(images, dialogue)

    # Tokenize
    text_ids = model.tokenizer.encode(sequence, return_tensors="pt")

    # Process images
    transform = model.transform
    images_enc = []
    images_dec = []
    for img in images[:num_images_used]:
        img_enc, img_dec = transform(img)
        images_enc.append(img_enc)
        images_dec.append(img_dec)

    # Stack images
    images_enc = torch.stack(images_enc).unsqueeze(0)  # (1, N, 3, 224, 224)
    images_dec = torch.stack(images_dec).unsqueeze(0)  # (1, N, 3, 512, 512)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=text_ids.to(model.device),
            image_tensors=images_enc.to(model.device),
            image_tensors_dec=images_dec.to(model.device),
            num_image_per_seq=torch.tensor([num_images_used]),
            max_new_tokens=50,
            num_beams=3,
            temperature=1.0
        )

    # Decode
    response = model.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return response
```

---

## MMFS Multi-Image Attention

**File:** `uni_interleaved/models/utils/ops/modules/mmfs.py:145-210`

```python
# MMFS handles multiple images via deformable attention
# Each text token can attend to all images

# Example: 3 images, 77 tokens each = 231 visual tokens
vis_embed = torch.cat([
    vis_embed_img1,  # (1, 77, 4096)
    vis_embed_img2,  # (1, 77, 4096)
    vis_embed_img3,  # (1, 77, 4096)
], dim=1)  # (1, 231, 4096)

# MMFS attention
# Query: text tokens (1, 512, 4096)
# Key/Value: visual tokens (1, 231, 4096)
# Output: text tokens with multi-image context (1, 512, 4096)

out = mmfs_attention(
    query=text_embed,
    value=vis_embed,
    spatial_shapes=[(8, 8), (8, 8), (8, 8)],  # 3 images, same resolution
    num_images=3
)
```

---

## Image Editing Workflow

```python
# Example: Remove object from image
original_image = load_image("cat_on_mat.jpg")

# Step 1: User provides original image
dialogue = [
    {"role": "user", "content": "Please remove the cat from this image."}
]

# Step 2: Generate edited image
edited_image = model.generate_images(
    images=[original_image],
    prompt="A mat without a cat",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Step 3: User asks about result
dialogue.append({"role": "assistant", "content": "[Generated image]"})
dialogue.append({"role": "user", "content": "Did you remove the cat?"})

# Step 4: Model responds
answer = model.generate_texts_interleaved(
    images=[original_image, edited_image],
    dialogue=dialogue,
    max_new_tokens=20
)
# Output: "Yes, I removed the cat from the image."
```

---

## Visual Dialogue

```python
# Example: Multi-turn VQA
image = load_image("street_scene.jpg")

dialogue_history = []

# Turn 1
dialogue_history.append({
    "role": "user",
    "content": "How many cars are in the image?"
})

response = model.generate_texts_interleaved(
    images=[image],
    dialogue=dialogue_history,
    max_new_tokens=10
)
dialogue_history.append({"role": "assistant", "content": response})
print(f"Assistant: {response}")  # "There are 3 cars."

# Turn 2
dialogue_history.append({
    "role": "user",
    "content": "What color is the first car?"
})

response = model.generate_texts_interleaved(
    images=[image],
    dialogue=dialogue_history,
    max_new_tokens=10
)
dialogue_history.append({"role": "assistant", "content": response})
print(f"Assistant: {response}")  # "Red."

# Turn 3
dialogue_history.append({
    "role": "user",
    "content": "Is it parked or moving?"
})

response = model.generate_texts_interleaved(
    images=[image],
    dialogue=dialogue_history,
    max_new_tokens=10
)
print(f"Assistant: {response}")  # "Parked."
```

---

## Handling Long Contexts

```python
# Problem: Too many images × 77 tokens = exceeds max length

# Solution 1: Sliding window
MAX_IMAGES = 10
if len(images) > MAX_IMAGES:
    # Keep only recent images
    images = images[-MAX_IMAGES:]
    dialogue = dialogue[-(MAX_IMAGES * 2):]  # Adjust dialogue too

# Solution 2: Compress dialogue history
def compress_dialogue(dialogue, max_turns=5):
    """Keep first turn + last N turns"""
    if len(dialogue) <= max_turns + 1:
        return dialogue

    return [dialogue[0]] + dialogue[-(max_turns):]

dialogue = compress_dialogue(dialogue, max_turns=5)
```

---

## Batched Interleaved Generation

```python
# Process multiple dialogues in parallel
def batch_interleaved_generation(model, batch_data):
    """
    batch_data: List of (images, dialogue) tuples
    """
    batch_sequences = []
    batch_images_enc = []
    batch_images_dec = []
    num_images_per_seq = []

    for images, dialogue in batch_data:
        # Format sequence
        sequence, num_imgs = format_interleaved_sequence(images, dialogue)
        batch_sequences.append(sequence)

        # Process images
        imgs_enc, imgs_dec = [], []
        for img in images[:num_imgs]:
            enc, dec = model.transform(img)
            imgs_enc.append(enc)
            imgs_dec.append(dec)

        batch_images_enc.append(torch.stack(imgs_enc))
        batch_images_dec.append(torch.stack(imgs_dec))
        num_images_per_seq.append(num_imgs)

    # Tokenize
    text_ids = model.tokenizer(
        batch_sequences,
        padding="longest",
        return_tensors="pt"
    ).input_ids

    # Pad images to max number in batch
    max_images = max(num_images_per_seq)
    for i in range(len(batch_images_enc)):
        pad_count = max_images - num_images_per_seq[i]
        if pad_count > 0:
            # Pad with zeros
            batch_images_enc[i] = F.pad(
                batch_images_enc[i],
                (0, 0, 0, 0, 0, 0, 0, pad_count)
            )
            batch_images_dec[i] = F.pad(
                batch_images_dec[i],
                (0, 0, 0, 0, 0, 0, 0, pad_count)
            )

    images_enc = torch.stack(batch_images_enc)
    images_dec = torch.stack(batch_images_dec)

    # Generate
    outputs = model.generate(
        input_ids=text_ids,
        image_tensors=images_enc,
        image_tensors_dec=images_dec,
        num_image_per_seq=torch.tensor(num_images_per_seq),
        max_new_tokens=50
    )

    # Decode
    responses = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses
```

---

## Training with Interleaved Data

**File:** `uni_interleaved/custom_datasets/train/mmc4_wds.py:150-220`

```python
# MMC4 dataset has interleaved image-text sequences
# Example sample:
{
    "images": [img1, img2, img3],
    "texts": ["Caption 1", "Caption 2", "Caption 3"]
}

# Collated sequence:
# "<bos> <|soi|> <|image|>×77 Caption 1 <|soi|> <|image|>×77 Caption 2 <|soi|> <|image|>×77 Caption 3 <eos>"

# Loss is computed on all text tokens (including between images)
```

---

## Common Issues

### 1. Image Order Confusion

```python
# Problem: Model confuses "first" vs "second" image

# Solution: Use explicit position markers
sequence = (
    "<bos> Image 1: <|soi|> <|image|>×77 "
    "Image 2: <|soi|> <|image|>×77 "
    "Question: What is in Image 1? Answer:"
)
```

### 2. Memory Overflow

```python
# Problem: Too many images × 77 tokens

# Solution: Limit images per sequence
MAX_IMAGES_PER_SEQ = 10
if len(images) > MAX_IMAGES_PER_SEQ:
    images = images[:MAX_IMAGES_PER_SEQ]
```

### 3. Slow Generation

```python
# Problem: Multi-image MMFS attention is expensive

# Solution: Use KV cache
output_ids = model.generate(
    ...,
    use_cache=True,  # Cache key/value tensors
    max_new_tokens=50
)

# Speedup: 2-3× faster for long sequences
```

---

## Summary

**Format:** `<|soi|> <|image|>×77 text1 <|soi|> <|image|>×77 text2 ...`
**Max images:** ~10 images (770 tokens)
**Use cases:** Visual dialogue, image editing, multi-image VQA

---

**Last Updated:** 2025-11-24
