import json
import requests
import os
import glob
import time
from datetime import datetime
import gradio as gr
from pathlib import Path
from image_downloader import resolve_online_collection

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
    with open(workflow_json, "r", encoding="utf-8") as f:
        workflow = json.load(f)
        params = workflow_definitions[workflow_name]["parameters"]
        arg_index = 0  # Add a separate counter for the args tuple
        for i, path in enumerate(params.values()):
            keys = path.strip("[]").split("][")
            keys = [key.strip('"') for key in keys]
            sub_dict = workflow
            for key in keys[:-1]:
                sub_dict = sub_dict[key]
            if "image_path" in params:
                print(args)
                if arg_index + 1 < len(
                    args
                ):  # Check if arg_index+1 is less than the length of args
                    if args[arg_index] == "Nilor Collection Name":
                        print(f"Resolving online collection: {args[arg_index+1]}")
                        path = resolve_online_collection(
                            args[arg_index + 1],
                            int(args[arg_index + 2]),
                            args[arg_index + 3],
                        )
                        image_count = count_images(path)
                        print(f"Detected {image_count} images in the collection.")
                        sub_dict[keys[-1]] = path
                        arg_index += (
                            4  # Increment the counter by 4 if it's a collection
                        )
                    else:
                        print(
                            f"Loading images from local directory: {args[arg_index+1]}"
                        )
                        path = args[arg_index + 1]
                        image_count = count_images(path)
                        print(f"Detected {image_count} images in the collection.")
                        sub_dict[keys[-1]] = path
                        arg_index += 2  # Increment the counter by 2 if it's a directory
            else:
                sub_dict[keys[-1]] = args[arg_index]
                arg_index += 1  # Increment the counter by 1 for other parameters
        # Rest of the function...
        # Rest of the function...]
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
            components.append(
                gr.Number(label="Max Images (only for Nilor Collection)", value=4)
            )
            components.append(
                gr.Checkbox(
                    label="Shuffle Images (only for Nilor Collection)", value=True
                )
            )
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
