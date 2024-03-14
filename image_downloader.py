import requests
from PIL import Image
from io import BytesIO
import os
import json

with open("SECRET_KEYS.json") as f:
    KEYS = json.load(f)

with open("config.json") as f:
    config = json.load(f)

IN_DIR = config["COMFY_ROOT"] + "input/WorkFlower/"


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c.isspace())


def resolve_online_collection(collection_name):
    try:
        # Define the API key and URL
        apiKey = KEYS["NILOR_API_KEY"]
        url = (
            f"{KEYS['NILOR_API_URI']}/collection/api-get-collection-image-urls-by-name"
        )

        # Define the headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {apiKey}",
        }

        # Define the data
        data = {"collectionName": collection_name}

        # Make the POST request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception if the request failed

        # Get the image URLs from the response
        image_urls = response.json()["imageUrls"]

        # Create a directory for the images
        sanitized_name = sanitize_name(collection_name)
        directory = os.path.join(IN_DIR, sanitized_name)
        os.makedirs(directory, exist_ok=True)

        # Download and save each image
        for i, image_url in enumerate(image_urls):
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception if the request failed
            image = Image.open(BytesIO(response.content))
            image.save(os.path.join(directory, f"image_{i}.png"))

        # Return the path to the directory
        return directory

    except Exception as e:
        print(f"Failed to resolve online collection: {e}")
        return None
