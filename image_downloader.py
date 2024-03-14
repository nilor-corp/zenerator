import requests
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime

with open("SECRET_KEYS.json") as f:
    KEYS = json.load(f)

with open("config.json") as f:
    config = json.load(f)

IN_DIR = config["COMFY_ROOT"] + "input/WorkFlower/"


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c.isspace())


def resize_image(image, base_width):
    # Calculate the aspect ratio
    w_percent = base_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))

    # Resize the image
    image = image.resize((base_width, h_size), Image.ANTIALIAS)
    return image


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
        print(f"Requesting image URLs for collection: {collection_name}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception if the request failed

        # Get the image URLs from the response
        image_urls = response.json()["imageUrls"]
        print(f"Received {len(image_urls)} image URLs")

        # Create a directory for the images
        sanitized_name = sanitize_name(collection_name)
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for images: {directory}")

        # Download and save each image
        for i, image_url in enumerate(image_urls):
            print(f"Downloading image {i+1} of {len(image_urls)}")
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception if the request failed
            image = Image.open(BytesIO(response.content))

            # Resize the image before saving
            image = resize_image(image, 512)

            image.save(os.path.join(directory, f"image_{i}.png"))
            print(f"Saved image {i+1} to {directory}")

        # Return the path to the directory
        print(f"Finished downloading images for collection: {collection_name}")
        return directory

    except Exception as e:
        print(f"Failed to resolve online collection: {e}")
        return None
