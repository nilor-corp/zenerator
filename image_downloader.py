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
IN_DIR = os.path.join(COMFY_ROOT, "input", "Zenerator")


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


def resolve_online_collection(
    collection_name, max_images=None, shuffle=False, progress=None
):
    # Reset progress
    if progress is not None:
        progress(0, desc="Requesting image URLs...")

    apiKey = os.environ["NILOR_API_KEY"]
    url = f"{os.environ['NILOR_API_URI']}/collection/api-get-collection-image-urls-by-name"
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
    downloaded = 0  # Counter for downloaded images

    if max_images:
        print(f"Selected first {max_images} image URLs")
        image_urls = image_urls[:max_images]
        num_images = max_images

    if progress is not None:
        progress(0.01, desc=f"Downloading {num_images} images...")

    if shuffle:
        random.shuffle(image_urls)

    sanitized_name = sanitize_name(collection_name)
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory for images: {directory}")

    if progress is not None:
        progress(0.02, desc=f"Downloading {len(image_urls)} images...")

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                download_image, i, url, directory, max_images, num_images
            ): i
            for i, url in enumerate(image_urls)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
                downloaded += 1
                if progress is not None:
                    progress(
                        0.1 + 0.8 * (downloaded / num_images),
                        desc=f"Downloaded {downloaded}/{num_images} images",
                    )
            except Exception as e:
                print(f"An error occurred while downloading image {i}: {e}")

    print(f"Finished downloading images for collection: {collection_name}")
    if progress is not None:
        progress(1.0, desc="Download complete!")
    return directory


def process_files(
    files, destination, input_type, singular=False, reorganising=False, source=None
):
    """
    Process files by copying them to the destination with proper naming.
    Args:
        files: List of files or single file path
        destination: Destination directory
        input_type: "images" or "image" or "video"
        singular: Whether this is a single file operation
        reorganising: Whether this is a reorganisation of already copied files
        source: Source directory (optional)
    """
    print(f"\nProcessing files: {'Reorganising' if reorganising else 'Organising'}")

    try:
        if singular:
            print("Processing single file")
            file = files[0] if isinstance(files, list) else files
            src = os.path.abspath(file)
            print(f"Source: {src}")

            ext = os.path.splitext(file)[1]
            new_filename = f"{input_type}_0000{ext}"
            dst = os.path.join(destination, new_filename)
            dst = os.path.abspath(dst)

            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied file to {dst}")
                return dst  # Return the path to the actual file for single files
            else:
                print(f"Source file not found: {src}")
                return None

        else:
            # Existing multiple files handling code
            filenames = files
            for i, filename in enumerate(filenames):
                print(f"Copying file {i+1} of {len(filenames)}")
                if source:
                    src = os.path.join(source, filename)
                else:
                    src = filename

                src = os.path.abspath(src)
                print(f"Source: {src}")
                ext = os.path.splitext(filename)[1]
                index_as_str = format(i, "04")
                new_filename = f"{input_type}_{index_as_str}{ext}"
                dst = os.path.join(destination, new_filename)
                dst = os.path.abspath(dst)

                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"Copied file {i+1} to {destination}")
                else:
                    print(f"Source file not found: {src}")
                    return None

            return os.path.abspath(destination)

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None


def process_video_file(file_path):
    try:
        print("\nProcessing video file")
        print(f"Video file path: {file_path}")

        if not os.path.isfile(file_path):
            print(f"Video file does not exist: {file_path}")
            return None

        # Create destination directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = os.path.splitext(os.path.basename(file_path))[0]
        destination = os.path.join(IN_DIR, folder_name, timestamp)
        os.makedirs(destination, exist_ok=True)
        print(f"Created directory for video: {destination}")

        # Copy video file to destination
        dst = os.path.join(destination, os.path.basename(file_path))
        shutil.copy(file_path, dst)
        print(f"Copied video file to {dst}")

        return os.path.abspath(dst)  # Return the full path to the copied video file
    except Exception as e:
        print(f"Failed to process video file: {e}")
        return None


def organise_local_files(
    filepath, input_type, max_images=None, shuffle=False, reorganising=False
):
    """
    Organise local files based on input type.
    Args:
        filepath: Path to file or directory
        input_type: "images" or "image" or "video"
        max_images: Maximum number of images to process (only for "images" type)
        shuffle: Whether to shuffle the files (only for "images" type)
        reorganising: Whether this is a reorganisation of already copied files
    """
    try:
        print("\nOrganising local files")
        filepath = os.path.abspath(filepath)  # Convert to absolute path
        print(f"Processing path: {filepath}")

        if input_type == "images":
            # Handle directory of images
            if not os.path.isdir(filepath):
                raise ValueError(f"Path is not a directory: {filepath}")

            print(f"Sorting files in directory: {filepath}")
            files = []
            for file in os.listdir(filepath):
                if file.lower().endswith(tuple(Image.registered_extensions().keys())):
                    files.append(file)

            files.sort()
            print(f"Found {len(files)} image files")

            if shuffle:
                random.shuffle(files)

            if max_images:
                files = files[:max_images]

            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            folder_name = os.path.basename(filepath)
            destination = os.path.join(IN_DIR, folder_name, timestamp)
            os.makedirs(destination, exist_ok=True)

            print(f"Created directory for images: {destination}")
            result = process_files(
                files,
                destination,
                input_type,
                reorganising=reorganising,
                source=filepath,
            )
            return result

        elif input_type == "image":
            # Handle single image file
            if not os.path.isfile(filepath):
                raise ValueError(f"Path is not a file: {filepath}")

            # Verify it's an image file
            if not filepath.lower().endswith(
                tuple(Image.registered_extensions().keys())
            ):
                raise ValueError(f"File is not a supported image type: {filepath}")

            # Create a timestamped directory for the single image
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            folder_name = os.path.splitext(os.path.basename(filepath))[0]
            destination = os.path.join(IN_DIR, folder_name, timestamp)
            os.makedirs(destination, exist_ok=True)

            print(f"Created directory for single image: {destination}")

            # Copy the file with a standardized name
            result = process_files(
                [filepath],
                destination,
                input_type,
                singular=True,
                reorganising=reorganising,
            )
            return result

        elif input_type == "video":
            return process_video_file(filepath)

    except Exception as e:
        print(f"Failed to reorganise local files: {str(e)}")
        return None


def copy_uploaded_files_to_local_dir(files, input_type, max_files=None, shuffle=False):
    """
    Args:
        files: list of uploaded files (could be list of tuples from Gradio upload component)
        For single image, 'files' will be a single filepath string
    """
    try:
        print("\nCopying uploaded files to local directory")
        print(f"Received files: {files}")
        print(f"Input type: {input_type}")

        # Create a new directory in the ComfyUI input folder
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, "uploaded", current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for {input_type}: {directory}")

        if input_type == "images":
            # Handle multiple images
            file_paths = [
                file[0] if isinstance(file, tuple) else file for file in files
            ]
            process_files(
                file_paths, directory, input_type, singular=False, reorganising=False
            )
            return organise_local_files(
                directory, input_type, max_files, shuffle, reorganising=True
            )
        else:  # For singular file inputs (image or video)
            # Handle single file
            file_path = files[0] if isinstance(files, list) else files
            if isinstance(file_path, tuple):
                file_path = file_path[0]
            print(f"Processing single file: {file_path}")
            return process_files(
                [file_path], directory, input_type, singular=True, reorganising=False
            )

    except Exception as e:
        print(f"Failed to copy local files: {str(e)}")
        return None
