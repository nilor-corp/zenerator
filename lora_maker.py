
import os
import shutil
import subprocess
from image_downloader import resolve_online_collection
import sys
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import random

anime_packages = [
            "accelerate==0.15.0",
            "diffusers[torch]==0.10.2",
            "einops==0.6.0",
            "tensorflow",
            "transformers",
            "safetensors",
            "huggingface-hub",
            "torchvision",
            "albumentations",
        ]
photo_packages = [
            "timm==0.6.12",
            "fairscale==0.4.13",
            "transformers==4.26.0",
            "requests==2.28.2",
            "accelerate==0.15.0",
            "diffusers[torch]==0.10.2",
            "einops==0.6.0",
            "safetensors==0.2.6",
        ]

def prepare_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    return directory

def print_sample_captions(dataset_folder, num_samples=5):
    caption_files = [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
    num_samples = min(num_samples, len(caption_files))
    sample_files = random.sample(caption_files, num_samples)

    for file in sample_files:
        with open(os.path.join(dataset_folder, file), 'r') as f:
            print(f"Caption for {file}: {f.read()}")

def ensure_packages(packages):
    """
    Ensures a list of Python packages are installed. If not, installs them.
    :param packages: A list of package names, optionally with versions.
    """
    for package in packages:
        try:
            pkg_resources.require(package)
            print(f"{package} is installed.")
        except (DistributionNotFound, VersionConflict):
            print(f"{package} not found or version conflict. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}. Error: {e}")

def tag_dataset(dataset_folder, method="Photos", tag_threshold=0.35, blacklist_tags="", caption_min=10, caption_max=75):
    
    if method != "Photos":
        ensure_packages(anime_packages)
    else:
        ensure_packages(photo_packages)  
    # Tagging logic goes here
    print(f"Tagging images in {dataset_folder} with method {method}")

def generate_lora(collection_name):
    print(f"Generate Lora with {collection_name}")
    path_to_images = resolve_online_collection(collection_name=collection_name)

    lora_folder = os.path.join(".", "loras", collection_name)
    dataset_folder = os.path.join(lora_folder, "dataset")
    output_folder = os.path.join(lora_folder, "output")

    prepare_directory(lora_folder)
    prepare_directory(dataset_folder)
    prepare_directory(output_folder)

    for filename in os.listdir(path_to_images):
        if filename.endswith((".jpg", ".png")):
            src = os.path.join(path_to_images, filename)
            dst = os.path.join(dataset_folder, filename)
            shutil.move(src, dst)

    print(f"Moved all images to {dataset_folder}")
    tag_dataset(dataset_folder)
    print(f"Finished tagging images in {dataset_folder}")
    print_sample_captions(dataset_folder)
