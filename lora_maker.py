import os
import shutil
import subprocess
from image_downloader import resolve_online_collection
import sys
import pkg_resources
import random
from packaging import version
from datetime import datetime


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
        with open(os.path.join(dataset_folder, file), "r") as f:
            print(f"Caption for {file}: {f.read()}")


def is_installed(package, required_version=None):
    print(f"Checking if {package} is installed...")
    try:
        installed_version = pkg_resources.get_distribution(package).version
        print(f"Found {package} version {installed_version}")
        if required_version is not None and version.parse(
            installed_version
        ) != version.parse(required_version):
            print(
                f"Found {package} version {installed_version}, but version {required_version} is required"
            )
            return False
        else:
            return True
    except pkg_resources.DistributionNotFound:
        print(f"{package} distribution not found")
        return False


def tag_dataset(
    dataset_folder,
    method="Photos",
    tag_threshold=0.35,
    blacklist_tags="",
    caption_min=10,
    caption_max=75,
):
    # clone the necessary scripts from GitHub
    kohya = os.path.join(os.getcwd(), "kohya-trainer")
    if not os.path.exists(kohya):
        subprocess.run(
            ["git", "clone", "https://github.com/kohya-ss/sd-scripts", kohya]
        )
        os.chdir(kohya)
        subprocess.run(
            ["git", "reset", "--hard", "9a67e0df390033a89f17e70df5131393692c2a55"]
        )
        os.chdir(os.getcwd())
    sys.path.append(kohya)

    # tag images or make captions
    if "anime" in method:
        # install dependencies
        dependencies = [
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
        for package in dependencies:
            if "==" in package:
                package_name, required_version = package.split("==")
            else:
                package_name = package
                required_version = None

            if not is_installed(package_name, required_version):
                print(f"Installing {package}...")
                try:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{package_name}=={required_version}"
                            if required_version
                            else package_name,
                            "--upgrade",
                            "--force-reinstall",
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {package}: {e.output}")
                    print("Trying to continue anyway...")
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(kohya, "finetune", "tag_images_by_wd14_tagger.py"),
                dataset_folder,
                "--repo_id=SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                "--model_dir=" + os.getcwd(),
                "--thresh=" + str(tag_threshold),
                "--batch_size=8",
                "--caption_extension=.txt",
                "--force_download",
                "--blacklist_tags=" + blacklist_tags,
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
    else:
        # install dependencie
        dependencies = [
            "timm==0.6.12",
            "fairscale==0.4.13",
            "transformers==4.26.0",
            "requests==2.28.2",
            "accelerate==0.15.0",
            "diffusers[torch]==0.10.2",
            "einops==0.6.0",
            "safetensors==0.2.6",
        ]

        for package in dependencies:
            package_name, required_version = package.split("==")
            if not is_installed(package_name, required_version):
                print(f"Installing {package}...")
                try:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{package_name}=={required_version}"
                            if required_version
                            else package_name,
                            "--force-reinstall",
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {package}: {e.output}")
        # run the captioning script
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(kohya, "finetune", "make_captions.py"),
                dataset_folder,
                "--beam_search",
                "--max_data_loader_n_workers=2",
                "--batch_size=8",
                "--min_length=" + str(caption_min),
                "--max_length=" + str(caption_max),
                "--caption_extension=.txt",
                "--blacklist_tags=" + blacklist_tags,
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)


def generate_lora(collection_name):
    print(f"Generate Lora with {collection_name}")
    # download all images in the collection into a directory
    path_to_images = resolve_online_collection(collection_name=collection_name)

    # make a folder for the Lora with collection_name relative to the current directory something like ./loras/collection_name
    lora_folder = os.path.join(".", "loras", collection_name)
    os.makedirs(lora_folder, exist_ok=True)

    # make two subfolders in the Lora folder called "dataset" and "output"
    dataset_folder = os.path.join(lora_folder, "dataset")
    output_folder = os.path.join(lora_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    # Prepare the dataset directory
    dataset_folder = prepare_directory(dataset_folder)

    # move all the images from path_to_images to the "dataset" folder
    for filename in os.listdir(path_to_images):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            src = os.path.join(path_to_images, filename)
            dst = os.path.join(dataset_folder, filename)
            if os.path.exists(dst):
                os.remove(dst)  # remove the file if it already exists
            shutil.move(src, dst)

    print(f"Moved all images to {dataset_folder}")
    print(f"Start tagging images in {dataset_folder}")
    # tag or caption images
    tag_dataset(dataset_folder)

    print(f"Finished tagging images in {dataset_folder}")
    # print some sample captions
    print_sample_captions(dataset_folder)
