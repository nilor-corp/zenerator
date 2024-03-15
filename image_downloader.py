import requests
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime
import random

with open("SECRET_KEYS.json") as f:
    KEYS = json.load(f)

with open("config.json") as f:
    config = json.load(f)

IN_DIR = config["COMFY_ROOT"] + "input/WorkFlower/"


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c.isspace())


def resize_image(image, base_width):
    w_percent = base_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.LANCZOS)
    return image


def resolve_online_collection(collection_name, max_images=None, shuffle=False):
    try:
        apiKey = KEYS["NILOR_API_KEY"]
        url = (
            f"{KEYS['NILOR_API_URI']}/collection/api-get-collection-image-urls-by-name"
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

        if max_images is not None and shuffle:
            image_urls = random.sample(image_urls, min(max_images, len(image_urls)))
            print(f"Randomly selected {len(image_urls)} image URLs")

        sanitized_name = sanitize_name(collection_name)
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for images: {directory}")

        for i, image_url in enumerate(image_urls):
            if max_images is not None and i >= max_images:
                print(f"Reached the limit of {max_images} images. Stopping download.")
                break

            print(f"Downloading image {i+1} of {len(image_urls)}")
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image = resize_image(image, 768)
            image.save(os.path.join(directory, f"image_{i}.png"))
            print(f"Saved image {i+1} to {directory}")

        print(f"Finished downloading images for collection: {collection_name}")
        return directory

    except Exception as e:
        print(f"Failed to resolve online collection: {e}")
        return None
