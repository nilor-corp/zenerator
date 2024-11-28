import requests
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

load_dotenv()

with open("config.json") as f:
    config = json.load(f)


COMFY_ROOT = os.path.normpath(config["COMFY_ROOT"])
IN_DIR = os.path.join(COMFY_ROOT, "input", "WorkFlower")


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c.isspace())


def resize_image(image, base_width):
    w_percent = base_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.LANCZOS)
    return image


def download_image(i, image_url, directory, max_images, num_images):
    if max_images is not None and i >= max_images:
        print(f"Reached the limit of {max_images} images. Stopping download.")
        return

    print(f"Downloading image {i+1} of {num_images}")
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image = resize_image(image, 768)
    image.save(os.path.join(directory, f"image_{i}.png"))
    print(f"Saved image {i+1} to {directory}")


def resolve_online_collection(collection_name, max_images=None, shuffle=False):
    try:
        apiKey = os.environ["NILOR_API_KEY"]
        url = (
            f"{os.environ['NILOR_API_URI']}/collection/api-get-collection-image-urls-by-name"
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {apiKey}",
        }
        data = {"collectionName": collection_name}

        print(f"Requesting image URLs for collection: {collection_name}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        image_urls = response.json()["imageUrls"]
        print(f"Received {len(image_urls)} image URLs")

        num_images = len(image_urls)

        if max_images is not None:
            num_images = min(max_images, num_images)
            if shuffle:
                image_urls = random.sample(image_urls, min(max_images, len(image_urls)))
                print(f"Randomly selected {len(image_urls)} image URLs")
            else:
                image_urls = image_urls[:max_images]
                print(f"Selected first {len(image_urls)} image URLs")

        sanitized_name = sanitize_name(collection_name)
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for images: {directory}")

        # print(image_urls)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    download_image, i, url, directory, max_images, num_images
                )
                for i, url in enumerate(image_urls)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

        print(f"Finished downloading images for collection: {collection_name}")
        return directory

    except Exception as e:
        print(f"Failed to resolve online collection: {e}")
        return None


def process_files(files, destination, input_type, singular=False, reorganising=False, source=None):
    print(f"\nProcessing files: {'Reorganising' if reorganising else 'Organising'}")
    if not singular:
        filenames = files          
        for i, filename in enumerate(filenames):
            print(f"Copying file {i+1} of {len(filenames)}")
            if source:
                src = os.path.join(source, filename)  # Use filename directly, not basename
            else:
                src = filename  # Use filename as is if no source directory
            
            src = os.path.abspath(src)
            print(f"Source: {src}")
            ext = os.path.splitext(filename)[1]  # Get extension from filename
            print(f"Extension: {ext}")
            index_as_str = format(i, "04")
            new_filename = f"{input_type}_{index_as_str}{ext}"
            print(new_filename)
            dst = os.path.join(destination, new_filename)
            dst = os.path.abspath(dst)  # Convert destination to absolute path
            
            if os.path.exists(src):  # Add existence check
                shutil.copy(src, dst)
                print(f"Copied file {i+1} to {destination}")
            else:
                print(f"Source file not found: {src}")
                return None
    else:
        print("We are using a File input")
        print(f"{files}")
        file = files if isinstance(files, str) else files[0]
        
        print(f"File: {file}")
        src = os.path.abspath(file)  # Convert to absolute path
        print(f"Source: {src}")
        ext = os.path.splitext(file)[1]
        dst = os.path.join(destination, f"{input_type}_0000{ext}")
        dst = os.path.abspath(dst)  # Convert destination to absolute path
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied file to {destination}")
        else:
            print(f"Source file not found: {src}")
            return None

    return os.path.abspath(destination)  # Return absolute path of destination directory



def organise_local_files(dir, input_type, max_images=None, shuffle=False, reorganising=False):
    try:
        print("\nOrganising local files")
        dir = os.path.abspath(dir)  # Convert input dir to absolute path
        print(f"Sorting files in directory: {dir}")
        
        files = []
        for file in os.listdir(dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                files.append(file)
        
        print(f"Sorted files: {files}")
        
        if not files:
            print("No image files found in directory")
            return None
            
        if shuffle:
            random.shuffle(files)
            
        if max_images:
            files = files[:max_images]
            
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = os.path.basename(dir)
        destination = os.path.join(IN_DIR, folder_name, timestamp)
        os.makedirs(destination, exist_ok=True)
        
        print(f"Created directory for images: {destination}")
        
        result = process_files(files, destination, input_type, reorganising=reorganising, source=dir)
        return os.path.abspath(result) if result else None  # Return absolute path

    except Exception as e:
        print(f"Failed to reorganise local files: {e}")
        return None
    



def copy_uploaded_files_to_local_dir(files, input_type, max_files=None, shuffle=False):
    """
    Args:
        files: list of uploaded files (could be list of tuples from Gradio upload component)
    """
    try:
        print("\nCopying uploaded files to local directory")

        # Extract file paths from the tuples
        file_paths = [file[0] if isinstance(file, tuple) else file for file in files]

        # Create a new directory in the ComfyUI input folder
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, "uploaded", current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for {input_type}: {directory}")

        if input_type == "images":
            process_files(file_paths, directory, input_type, singular=False, reorganising=False)
            return organise_local_files(directory, input_type, max_files, shuffle, reorganising=True)
        else:  # For singular file inputs
            return process_files(file_paths, directory, input_type, singular=True, reorganising=False)

    except Exception as e:
        print(f"Failed to copy local files: {e}")
        return None

