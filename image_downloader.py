import requests
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def reorganise_local_files(dir, max_images=None, shuffle=False):
    try:
        if not os.path.exists(dir):
            print(f"Directory does not exist: {dir}")
            return None

        print(f"Listing files in directory: {dir}")
        files = os.listdir(dir)
        if shuffle:
            random.shuffle(files)
            print(f"Shuffled files in directory {dir}")

        sanitized_name = sanitize_name(os.path.basename(dir))
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for images: {directory}")

        for i, file in enumerate(files):
            if max_images is not None and i >= max_images:
                print(f"Reached the limit of {max_images} images. Stopping copy.")
                break

            print(f"Copying file {i+1} of {len(files)}")
            src = os.path.join(dir, file)
            dst = os.path.join(directory, f"image_{i}.png")
            shutil.copy(src, dst)
            print(f"Copied file {i+1} to {directory}")

        print(f"Finished copying images from directory: {dir}")
        return directory

    except Exception as e:
        print(f"Failed to shuffle local files: {e}")
        return None
