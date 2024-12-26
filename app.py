import gradio as gr
import os
import json
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import torch
import clip
from PIL import Image
import shutil
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from pathlib import Path
import requests

# URL to the JSON file
url = "https://huggingface.co/datasets/subashpoudel/image-embeddings/resolve/main/image_embeddings.json"

# Download and save the file
response = requests.get(url)
if response.status_code == 200:
    with open("image_embeddings.json", "wb") as file:
        file.write(response.content)
    print("File downloaded as image_embeddings.json")
else:
    print(f"Failed to download the file. HTTP status code: {response.status_code}")


# Define paths
EMBEDDINGS_FILE = "image_embeddings.json"
destination_folder = "extracted_frames"
video_output_path = "final_generated_video.mp4"

# Moving all the images to content/images folder
dataset_path = "/root/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1"
output_dir = "/content/images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List image files from the dataset (check for multiple formats if needed)
image_paths = [str(p) for p in Path(dataset_path).rglob('*') if p.suffix.lower() in ['.jpeg', '.jpg', '.png', '.bmp', '.gif', '.tiff']]
 # Adjust file extension if needed

# Count the number of images
num_images = len(image_paths)
print(f"Total number of images found: {num_images}")

# Check if the dataset contains any images
if num_images == 0:
    print("No images found in the specified dataset path. Exiting.")
    # exit()

try:
    # Move all images to the output directory
    for img_path in image_paths:
        try:
            # Get the file name and destination path
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, img_name)

            # Move the image to the destination directory
            shutil.move(img_path, dest_path)
            # print(f"Moved: {img_name} to {output_dir}")
        except Exception as e:
            print(f"Error moving {img_path}: {e}")
            continue

    print(f"All images moved to: {output_dir}")

except Exception as e:
    print(f"Error during moving process: {e}")


# Load embeddings
def load_embeddings():
    with open(EMBEDDINGS_FILE, "r") as f:
        image_embeddings_dict = json.load(f)
    return image_embeddings_dict

image_embeddings_dict = load_embeddings()

# Initialize CLIP model and FAISS index
def initialize_model_and_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    image_paths = list(image_embeddings_dict.keys())
    image_embeddings = np.array(list(image_embeddings_dict.values())).squeeze()
    image_embeddings = normalize(image_embeddings, axis=1)
    dim = image_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(image_embeddings)
    return clip_model, faiss_index, image_paths, device

clip_model, faiss_index, image_paths, device = initialize_model_and_index()

# Function to get CLIP embedding for text
def get_text_embedding(query_text):
    text = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    return text_features.cpu().numpy()

# Retrieve images for a given query
def retrieve_images(query_text, k=5):
    query_embedding = get_text_embedding(query_text)
    query_embedding = normalize(query_embedding, axis=1)
    _, indices = faiss_index.search(query_embedding, k)
    return [image_paths[i] for i in indices[0]]

# Main video generation function
def generate_video(user_query):
    sentences = [sentence.strip() for sentence in user_query.strip().split('.') if sentence]
    retrieved_images = []

    # Collect images for each sentence
    for sentence in sentences:
        top_images = retrieve_images(sentence, k=1)
        retrieved_images.extend(top_images)

    # Save retrieved images to destination folder
    os.makedirs(destination_folder, exist_ok=True)
    for img_path in retrieved_images:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(destination_folder, filename)
        shutil.copy(img_path, dest_path)

    # Generate video from retrieved images using StableVideoDiffusion
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.manual_seed(42)
    final_frames = []

    for img_path in retrieved_images:
        image = load_image(img_path)
        image = image.resize((1024, 576))
        frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
        final_frames.extend(frames)

    export_to_video(final_frames, video_output_path, fps=10)

    # Return both the video path and the images to display
    return video_output_path, retrieved_images

# Gradio Interface
def retrieve_and_display_images(user_query):
    sentences = [sentence.strip() for sentence in user_query.strip().split('.') if sentence]
    retrieved_images = []

    # Collect images for each sentence
    for sentence in sentences:
        top_images = retrieve_images(sentence, k=1)
        retrieved_images.extend(top_images)

    # Convert image paths to PIL images for display
    pil_images = [Image.open(img_path) for img_path in retrieved_images]

    return pil_images  # Return PIL images for gallery display

def generate_video_from_retrieved_images(user_query):
    video_path, _ = generate_video(user_query)
    return video_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### Text-to-Video Generation App")
    gr.Markdown("Generate a video from natural language descriptions using image retrieval and AI-based video generation.")
    
    user_query = gr.Textbox(label="Enter your query:", placeholder="Create a short video of a bustling city street at night.")

    # Buttons for interacting
    with gr.Row():
        retrieve_button = gr.Button("Retrieve Images")
        generate_video_button = gr.Button("Generate Video")

    # Outputs
    retrieved_images_gallery = gr.Gallery(label="Retrieved Images", columns=3)
    generated_video_output = gr.Video(label="Generated Video")
    
    # Logic to update gallery with retrieved images when button is clicked
    retrieve_button.click(fn=retrieve_and_display_images, inputs=user_query, outputs=retrieved_images_gallery)
    generate_video_button.click(fn=generate_video_from_retrieved_images, inputs=user_query, outputs=generated_video_output)

demo.launch(share=True)
