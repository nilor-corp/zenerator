import json
import requests
import os
import glob
import time
from datetime import datetime
import gradio as gr
from pathlib import Path
from image_downloader import resolve_online_collection
from image_downloader import reorganise_local_files

with open("config.json") as f:
    config = json.load(f)

URL = config["COMFY_URL"]
OUT_DIR = config["COMFY_ROOT"] + "output/WorkFlower/"


def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    try:
        requests.post(URL, data=data)
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")


with open("workflow_definitions.json") as f:
    workflow_definitions = json.load(f)


def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [
        f for f in files if f.lower().endswith(("png", "jpg", "jpeg", "gif"))
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


def get_latest_video(folder):
    files = os.listdir(folder)
    video_files = [f for f in files if f.lower().endswith(("mp4", "mov"))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video


def count_images(directory):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"]
    image_count = sum(
        len(glob.glob(os.path.join(directory, ext))) for ext in extensions
    )
    return image_count


def run_workflow(workflow_name, *args):
    print("inside run workflow with args: " + str(args))
    workflow_json = (
        "./workflows/" + workflow_definitions[workflow_name]["filename"] + ".json"
    )
    # Open the workflow JSON file
    with open(workflow_json, "r", encoding="utf-8") as f:
        # Load the JSON data into the workflow variable
        workflow = json.load(f)
        # Get the parameters for the current workflow
        params = workflow_definitions[workflow_name]["parameters"]
        # Initialize a counter for the arguments tuple
        arg_index = 0
        # Iterate over the values in the parameters dictionary
        for i, path in enumerate(params.values()):
            # Split the path into keys, removing brackets and splitting on ']['
            keys = path.strip("[]").split("][")
            # Remove any leading or trailing quotes from the keys
            keys = [key.strip('"') for key in keys]
            # Start with the full workflow dictionary
            sub_dict = workflow
            # Traverse the dictionary using the keys, stopping before the last key
            for key in keys[:-1]:
                sub_dict = sub_dict[key]
            # If the parameters include an image path
            if "image_path" in params:
                # Print the arguments for debugging
                print(args)
                # Check if the next argument index is within the range of the arguments tuple
                if arg_index + 1 < len(args):
                    # If the current argument is a Nilor Collection Name
                    if args[arg_index] == "Nilor Collection Name":
                        # Print a message indicating that the code is resolving an online collection
                        print(f"Resolving online collection: {args[arg_index+1]}")
                        # Resolve the online collection and get the path to the images
                        path = resolve_online_collection(
                            args[arg_index + 1],
                            int(args[arg_index + 2]),
                            args[arg_index + 3],
                        )
                        # Count the number of images in the path
                        image_count = count_images(path)
                        # Print the number of images
                        print(f"Detected {image_count} images in the collection.")
                        # Set the last key in the sub-dictionary to the path
                        sub_dict[keys[-1]] = path
                        # Increment the argument index by 4 to skip the arguments used for the online collection
                        arg_index += 4
                    else:
                        # If the current argument is not a Nilor Collection Name, it's a local directory
                        # Print a message indicating that the code is loading images from a local directory
                        print(
                            f"Loading images from local directory: {args[arg_index+1]}"
                        )
                        # Get the path to the local directory
                        path = args[arg_index + 1]
                        # Count the number of images in the path
                        image_count = count_images(path)
                        # Print the number of images
                        print(f"Detected {image_count} images in the collection.")

                        # make a copy of the relevant local files, optionally shuffling them
                        path = reorganise_local_files(
                            path, int(args[arg_index + 2]), args[arg_index + 3]
                        )

                        # Set the last key in the sub-dictionary to the path
                        sub_dict[keys[-1]] = path
                        # Increment the argument index by 4 to skip the arguments used for the online collection
                        arg_index += 4
                else:
                    # If the current argument is not an image path, it's a regular parameter
                    # Set the last key in the sub-dictionary to the current argument
                    sub_dict[keys[-1]] = args[arg_index]
                    # Increment the argument index by 1 to move to the next argument
                    arg_index += 1
            # Rest of the function...
        current_datetime = datetime.now().strftime("%Y-%m-%d")
        output_directory = OUT_DIR
        previous_video = get_latest_video(output_directory)
        print(f"Previous video: {previous_video}")
        start_queue(workflow)
        try:
            while True:
                latest_video = get_latest_video(output_directory)
                if latest_video != previous_video:
                    print(f"New video created: {latest_video}")
                    time.sleep(1)
                    return latest_video
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return None


def run_workflow_with_name(workflow_name):
    def wrapper(*args):
        return run_workflow(workflow_name, *args)

    return wrapper


def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["filename"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None


def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")
    components = []
    for param in workflow_definitions[workflow_name]["parameters"]:
        label = workflow_definitions[workflow_name]["labels"].get(param, param)
        if param == "text_1" or param == "text_2":
            components.append(gr.Textbox(label=label))
        elif param == "images":
            components.append(gr.Files(label=label))
        elif param == "image_path":
            components.append(
                gr.Radio(
                    ["Local Directory", "Nilor Collection Name"],
                    label="Images Path Type",
                    value="Local Directory",
                )
            )
            components.append(gr.Textbox(label="Collection Name or Directory Path"))
            components.append(gr.Number(label="Max Images", value=4))
            components.append(gr.Checkbox(label="Shuffle Images", value=False))
        elif param == "video_file":
            components.append(gr.File(label=label))
        elif param == "video_path":
            components.append(gr.Textbox(label=label))
        elif param == "bool_1":
            components.append(gr.Checkbox(label=label))
    return components


with gr.Blocks(title="WorkFlower") as demo:
    with gr.Row():
        with gr.Column():
            tabs = gr.Tabs()
            with tabs:
                for workflow_name in workflow_definitions.keys():
                    with gr.TabItem(label=workflow_name):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    preview_gif = gr.Image(
                                        label="Preview GIF",
                                        value=update_gif(workflow_name),
                                    )
                                info = gr.Markdown(
                                    workflow_definitions[workflow_name].get("info", "")
                                )
                            with gr.Column():
                                run_button = gr.Button("Run Workflow")
                                components = create_tab_interface(workflow_name)
                            with gr.Column():
                                output_player = gr.Video(
                                    label="Output Video", autoplay=True
                                )
                                with gr.Row():
                                    # mark_bad = gr.Button("Mark as Bad")
                                    # mark_good = gr.Button("Mark as Good")
                                    upscale_button = gr.Button("Upscale")
                        run_button.click(
                            fn=run_workflow_with_name(workflow_name),
                            inputs=components,
                            outputs=[output_player],
                        )
    demo.launch(favicon_path="favicon.png")
