# Visual Question Answering Application

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time:** 30 minutes

---

## Overview

Build a complete VQA web application using DEEM with Gradio for the UI.

---

## Installation

```bash
pip install gradio==3.50.0
pip install pillow
pip install torch torchvision
```

---

## Basic VQA Interface

```python
import gradio as gr
import torch
from PIL import Image
from uni_interleaved.models.uni_interleaved import MMInterleaved

# Load model
def load_model(checkpoint_path):
    model = MMInterleaved.from_pretrained(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_model("./OUTPUT/sft_vqa/checkpoint-final")

# VQA function
def answer_question(image, question):
    """Answer a question about an image"""

    # Format prompt
    prompt = f"Question: {question} Answer:"

    # Generate answer
    with torch.no_grad():
        answer = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=20,
            num_beams=5,
            temperature=1.0
        )

    return answer.strip()

# Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question", placeholder="What is in this image?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="DEEM Visual Question Answering",
    description="Ask questions about images using DEEM",
    examples=[
        ["examples/cat.jpg", "What color is the cat?"],
        ["examples/street.jpg", "How many cars are visible?"],
        ["examples/food.jpg", "What type of food is this?"]
    ]
)

demo.launch(share=True)
```

---

## Advanced Interface with Multiple Tasks

```python
import gradio as gr

def process_image(image, task, question_or_prompt):
    """
    task: "vqa", "caption", or "generate"
    question_or_prompt: User input
    """

    if task == "vqa":
        # Visual Question Answering
        prompt = f"Question: {question_or_prompt} Answer:"
        output = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=20,
            num_beams=5
        )
        return output, None

    elif task == "caption":
        # Image Captioning
        output = model.generate_texts(
            images=image,
            prompt="",
            max_new_tokens=30,
            num_beams=5
        )
        return output, None

    elif task == "generate":
        # Image Generation
        output_image = model.generate_images(
            images=image,
            prompt=question_or_prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        return "", output_image

# Multi-task interface
with gr.Blocks() as demo:
    gr.Markdown("# DEEM Multimodal Assistant")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            task_selector = gr.Radio(
                ["vqa", "caption", "generate"],
                label="Task",
                value="vqa"
            )
            text_input = gr.Textbox(
                label="Question/Prompt",
                placeholder="Enter question or generation prompt"
            )
            submit_btn = gr.Button("Submit")

        with gr.Column():
            text_output = gr.Textbox(label="Text Output")
            image_output = gr.Image(label="Generated Image")

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, task_selector, text_input],
        outputs=[text_output, image_output]
    )

    gr.Examples(
        examples=[
            ["examples/cat.jpg", "vqa", "What color is the cat?"],
            ["examples/street.jpg", "caption", ""],
            ["examples/cat.jpg", "generate", "a dog in the same pose"]
        ],
        inputs=[image_input, task_selector, text_input]
    )

demo.launch(share=True)
```

---

## Chatbot Interface

```python
import gradio as gr

# Global state for conversation history
conversation_history = []

def chat_with_image(image, message, history):
    """Chat about an image with history"""

    # Build dialogue
    dialogue = []
    for user_msg, assistant_msg in history:
        dialogue.append({"role": "user", "content": user_msg})
        dialogue.append({"role": "assistant", "content": assistant_msg})

    # Add current message
    dialogue.append({"role": "user", "content": message})

    # Format prompt
    prompt = ""
    for turn in dialogue:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            prompt += f"User: {content} "
        else:
            prompt += f"Assistant: {content} "

    prompt += "Assistant:"

    # Generate response
    with torch.no_grad():
        response = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )

    # Update history
    history.append((message, response))

    return "", history

# Chatbot interface
with gr.Blocks() as demo:
    gr.Markdown("# DEEM Visual Chatbot")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation")
            message_input = gr.Textbox(
                label="Message",
                placeholder="Ask something about the image..."
            )
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear")

    send_btn.click(
        fn=chat_with_image,
        inputs=[image_input, message_input, chatbot],
        outputs=[message_input, chatbot]
    )

    clear_btn.click(
        fn=lambda: (None, []),
        outputs=[image_input, chatbot]
    )

demo.launch(share=True)
```

---

## REST API with FastAPI

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model = load_model("./OUTPUT/sft_vqa/checkpoint-final")

@app.post("/vqa")
async def vqa_endpoint(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """VQA endpoint"""

    # Load image
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Answer question
    prompt = f"Question: {question} Answer:"
    answer = model.generate_texts(
        images=pil_image,
        prompt=prompt,
        max_new_tokens=20,
        num_beams=5
    )

    return JSONResponse({
        "question": question,
        "answer": answer.strip()
    })

@app.post("/caption")
async def caption_endpoint(image: UploadFile = File(...)):
    """Captioning endpoint"""

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    caption = model.generate_texts(
        images=pil_image,
        prompt="",
        max_new_tokens=30,
        num_beams=5
    )

    return JSONResponse({
        "caption": caption.strip()
    })

# Run server
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### Client Usage

```python
import requests

# VQA request
files = {"image": open("cat.jpg", "rb")}
data = {"question": "What color is the cat?"}

response = requests.post(
    "http://localhost:8000/vqa",
    files=files,
    data=data
)

print(response.json())
# {"question": "What color is the cat?", "answer": "orange"}
```

---

## Mobile App (Streamlit)

```python
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="DEEM VQA",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DEEM Visual Question Answering")

# Sidebar
st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Max tokens", 10, 100, 20)
num_beams = st.sidebar.slider("Num beams", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)

# Main area
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    question = st.text_input("Question", "What is in this image?")

    if st.button("Submit"):
        if uploaded_file:
            with st.spinner("Generating answer..."):
                prompt = f"Question: {question} Answer:"
                answer = model.generate_texts(
                    images=image,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    temperature=temperature
                )

                st.session_state.answer = answer.strip()

with col2:
    st.header("Output")
    if "answer" in st.session_state:
        st.success("Answer:")
        st.write(st.session_state.answer)

# Run: streamlit run app.py
```

---

## Batch Processing

```python
import pandas as pd
from tqdm import tqdm

def batch_vqa(csv_file, output_file):
    """
    Process CSV with columns: image_path, question
    """
    df = pd.read_csv(csv_file)

    answers = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = Image.open(row["image_path"]).convert("RGB")
        question = row["question"]

        prompt = f"Question: {question} Answer:"
        answer = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=20,
            num_beams=5
        )

        answers.append(answer.strip())

    df["answer"] = answers
    df.to_csv(output_file, index=False)

# Usage
batch_vqa("questions.csv", "answers.csv")
```

---

## Performance Optimization

### 1. Model Quantization

```python
# Use int8 quantization for faster inference
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 2-3× faster, 75% less memory
```

### 2. Batch Processing

```python
# Process multiple questions about same image
def batch_questions(image, questions):
    answers = []
    for question in questions:
        prompt = f"Question: {question} Answer:"
        answer = model.generate_texts(
            images=image,
            prompt=prompt,
            max_new_tokens=20
        )
        answers.append(answer.strip())

    return answers

# Usage
image = Image.open("cat.jpg")
questions = [
    "What color is the cat?",
    "Where is the cat sitting?",
    "Is the cat sleeping?"
]
answers = batch_questions(image, questions)
```

---

## Summary

**Interface:** Gradio, Streamlit, FastAPI
**Features:** VQA, captioning, chatbot
**Deployment:** Web app, REST API, mobile

---

**Last Updated:** 2025-11-24
