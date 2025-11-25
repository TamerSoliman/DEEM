# VQA Evaluation

**Difficulty:** ⭐⭐⭐ Intermediate
**Time:** 30 minutes

---

## Overview

Evaluate DEEM on standard VQA benchmarks: VQAv2, TextVQA, GQA, and OK-VQA.

---

## VQAv2 Evaluation

**File:** `evaluate.py:150-220`

### Dataset Format

```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "What color is the cat?",
  "answers": [
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange and white", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "tan", "answer_confidence": "maybe"},
    {"answer": "orange", "answer_confidence": "yes"},
    {"answer": "orange", "answer_confidence": "yes"}
  ]
}
```

---

## Inference

```python
import torch
from PIL import Image

def evaluate_vqa(model, dataset, output_file="vqa_results.json"):
    model.eval()
    results = []

    for sample in tqdm(dataset):
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Format prompt
        question = sample["question"]
        prompt = f"Question: {question} Answer:"

        # Generate answer
        with torch.no_grad():
            answer = model.generate_texts(
                images=image,
                prompt=prompt,
                max_new_tokens=10,
                num_beams=5,  # Beam search for better quality
                temperature=1.0
            )

        # Clean answer
        answer = answer.strip().lower()
        answer = answer.rstrip(".")

        results.append({
            "question_id": sample["question_id"],
            "answer": answer
        })

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f)

    return results
```

---

## Scoring

### VQA Accuracy Metric

```python
def vqa_accuracy(prediction, ground_truths):
    """
    VQA accuracy: min(#humans_gave_answer / 3, 1.0)
    """
    # Count how many annotators gave this answer
    num_matches = sum(1 for gt in ground_truths if gt["answer"] == prediction)

    # VQA accuracy formula
    accuracy = min(num_matches / 3.0, 1.0)

    return accuracy

# Example
ground_truths = [
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange"},
    {"answer": "orange and white"},
    {"answer": "tan"},
    {"answer": "orange"}
]

prediction = "orange"
acc = vqa_accuracy(prediction, ground_truths)  # 8/3 → 1.0 (capped)

prediction = "tan"
acc = vqa_accuracy(prediction, ground_truths)  # 1/3 → 0.33
```

### Compute Overall Accuracy

```python
def compute_vqa_metrics(results, annotations):
    total_accuracy = 0.0

    for result in results:
        qid = result["question_id"]
        prediction = result["answer"]

        # Find ground truth
        gt = annotations[qid]
        accuracy = vqa_accuracy(prediction, gt["answers"])

        total_accuracy += accuracy

    overall_accuracy = 100 * total_accuracy / len(results)
    return overall_accuracy

# Usage
accuracy = compute_vqa_metrics(results, annotations)
print(f"VQAv2 Accuracy: {accuracy:.2f}%")
```

---

## TextVQA Evaluation

**File:** `evaluate.py:250-310`

### Format

```json
{
  "question_id": 12345,
  "image_id": "abcd1234",
  "question": "What is written on the sign?",
  "answers": ["STOP", "stop", "Stop"]
}
```

### OCR-Aware Prompting

```python
# TextVQA requires reading text in images
prompt = f"Read the text in the image carefully. Question: {question} Answer:"

answer = model.generate_texts(
    images=image,
    prompt=prompt,
    max_new_tokens=20,  # Longer for multi-word text
    num_beams=5
)
```

### Accuracy Metric

```python
def textvqa_accuracy(prediction, ground_truths):
    """
    Exact match or fuzzy match
    """
    prediction = prediction.lower().strip()

    # Check exact match
    if prediction in [gt.lower() for gt in ground_truths]:
        return 1.0

    # Check fuzzy match (edit distance)
    for gt in ground_truths:
        if edit_distance(prediction, gt.lower()) <= 2:
            return 0.5

    return 0.0
```

---

## GQA Evaluation

**File:** `evaluate.py:320-380`

### Format

```json
{
  "questionId": "12345",
  "imageId": "n12345",
  "question": "Is there a cat in the image?",
  "answer": "yes",
  "fullAnswer": "Yes, there is a cat."
}
```

### Binary Questions

```python
# GQA has many yes/no questions
# Post-process answers
def normalize_answer(answer):
    answer = answer.lower().strip()

    # Map to yes/no
    if answer in ["yes", "yeah", "yep", "correct", "true"]:
        return "yes"
    if answer in ["no", "nope", "false", "incorrect"]:
        return "no"

    return answer

# Evaluate
prediction = normalize_answer(model_output)
ground_truth = sample["answer"]
correct = (prediction == ground_truth)
```

---

## OK-VQA Evaluation

**File:** `evaluate.py:390-440`

### Knowledge-Based Questions

```json
{
  "question_id": 123,
  "question": "What country is this flag from?",
  "answers": ["France", "france", "french"]
}
```

### External Knowledge Prompting

```python
# OK-VQA requires world knowledge
prompt = (
    "Use your knowledge to answer the following question. "
    f"Question: {question} Answer:"
)

answer = model.generate_texts(
    images=image,
    prompt=prompt,
    max_new_tokens=15,
    temperature=0.7  # Higher temperature for creative answers
)
```

---

## Batch Inference

```python
# Process multiple samples in parallel
def batch_evaluate(model, dataset, batch_size=32):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collator
    )

    results = []
    for batch in tqdm(dataloader):
        images = batch["images"]
        questions = batch["questions"]

        # Batch generation
        answers = model.generate_texts_batch(
            images=images,
            prompts=[f"Question: {q} Answer:" for q in questions],
            max_new_tokens=10,
            num_beams=5
        )

        for qid, answer in zip(batch["question_ids"], answers):
            results.append({
                "question_id": qid,
                "answer": answer.strip().lower()
            })

    return results
```

---

## Running Official Evaluation

### VQAv2

```bash
# Download evaluation code
git clone https://github.com/GT-Vision-Lab/VQA.git

# Generate results
python evaluate.py --dataset vqav2 --output vqa_results.json

# Compute accuracy
cd VQA/PythonEvaluationTools
python vqaEval.py \
    --result vqa_results.json \
    --annotation v2_mscoco_val2014_annotations.json \
    --question v2_OpenEnded_mscoco_val2014_questions.json
```

### TextVQA

```bash
# Download evaluation script
wget https://dl.fbaipublicfiles.com/textvqa/scripts/eval.py

# Run evaluation
python eval.py \
    --result_file textvqa_results.json \
    --annotation_file TextVQA_0.5.1_val.json
```

---

## Expected Performance

### DEEM Baseline

| Benchmark | Accuracy | Notes |
|-----------|----------|-------|
| VQAv2 | 66.8% | After Stage 2 SFT |
| TextVQA | 52.3% | Requires OCR |
| GQA | 58.1% | Spatial reasoning |
| OK-VQA | 44.2% | External knowledge |

### State-of-the-Art

| Model | VQAv2 | TextVQA |
|-------|-------|---------|
| DEEM | 66.8% | 52.3% |
| LLaVA-1.5 | 78.5% | 58.2% |
| GPT-4V | 87.1% | 78.0% |

---

## Debugging Low Accuracy

### 1. Inspect Predictions

```python
# Print sample predictions
for i in range(10):
    sample = dataset[i]
    prediction = results[i]["answer"]
    ground_truth = sample["answers"][0]["answer"]

    print(f"Q: {sample['question']}")
    print(f"Pred: {prediction}")
    print(f"GT: {ground_truth}")
    print(f"Correct: {prediction == ground_truth}")
    print()
```

### 2. Check Answer Distribution

```python
from collections import Counter

# Count predicted answers
pred_counter = Counter([r["answer"] for r in results])
print(pred_counter.most_common(20))

# Common issues:
# - Too many "yes" answers → model is biased
# - Many empty answers → generation failed
# - Too verbose → need better prompting
```

### 3. Visualize Failures

```python
# Find cases where model is wrong
failures = []
for result, sample in zip(results, dataset):
    if result["answer"] != sample["answers"][0]["answer"]:
        failures.append({
            "image": sample["image"],
            "question": sample["question"],
            "prediction": result["answer"],
            "ground_truth": sample["answers"][0]["answer"]
        })

# Show failures
for fail in failures[:10]:
    display(fail["image"])
    print(f"Q: {fail['question']}")
    print(f"Pred: {fail['prediction']}")
    print(f"GT: {fail['ground_truth']}")
```

---

## Summary

**Benchmarks:** VQAv2, TextVQA, GQA, OK-VQA
**Metric:** VQA accuracy (min(count/3, 1.0))
**DEEM Performance:** VQAv2 66.8%, TextVQA 52.3%

---

**Last Updated:** 2025-11-24
