import os
import urllib.request
import torch
import gradio as gr
from PIL import Image
from transformers import CLIPTokenizer

import model_loader
import pipeline

# -----------------------------
# Device selection
# -----------------------------
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
MERGES_PATH = os.path.join(DATA_DIR, "merges.txt")
CKPT_PATH = os.path.join(DATA_DIR, "v1-5-pruned-emaonly.ckpt")

# -----------------------------
# Download helper
# -----------------------------
def download_if_not_exists(url, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print(f"{os.path.basename(filepath)} already exists.")

# -----------------------------
# Official sources
# -----------------------------
VOCAB_URL = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json"
MERGES_URL = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt"
SD15_CKPT_URL = (
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/"
    "resolve/main/v1-5-pruned-emaonly.ckpt"
)

# -----------------------------
# Ensure required files exist
# -----------------------------
download_if_not_exists(VOCAB_URL, VOCAB_PATH)
download_if_not_exists(MERGES_URL, MERGES_PATH)
download_if_not_exists(SD15_CKPT_URL, CKPT_PATH)

# -----------------------------
# Load tokenizer and models ONCE
# -----------------------------
tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)

models = model_loader.preload_models_from_standard_weights(
    CKPT_PATH,
    DEVICE
)

# -----------------------------
# Inference function
# -----------------------------
def generate_image(
    prompt,
    steps=30,
    cfg_scale=8.0,
    seed=42
):
    if not prompt or prompt.strip() == "":
        return None

    with torch.no_grad():
        image = pipeline.generate(
            prompt=prompt,
            uncond_prompt="",
            input_image=None,
            strength=0.9,
            do_cfg=True,
            cfg_scale=cfg_scale,
            sampler_name="ddpm",
            n_inference_steps=steps,
            seed=int(seed),
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )

    return Image.fromarray(image)

# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate"),
        gr.Slider(10, 50, value=30, step=1, label="Inference Steps"),
        gr.Slider(1.0, 14.0, value=8.0, step=0.5, label="CFG Scale"),
        gr.Number(value=42, label="Seed"),
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion v1.5 — From Scratch",
    description=(
        "From-scratch PyTorch implementation of Stable Diffusion v1.5 inference. "
        "UNet, VAE, DDPM sampler, attention, CFG, and pipeline implemented manually. "
        "Pretrained v1.5 weights are downloaded at runtime and used only for inference."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
