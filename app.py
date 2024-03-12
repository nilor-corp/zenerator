import json
import requests
import os
import time
from datetime import datetime
import gradio as gr
from pathlib import Path

with open("config.json") as f:
    config = json.load(f)

# URL for the server that will process the workflows
URL = config["COMFY_URL"]
# Directory where the output of the workflows will be stored
OUT_DIR = config["OUT_DIR"]


# Function to start a workflow on the server
def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    try:
        requests.post(URL, data=data)
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
        # Here you can add code to retry the request, or handle the error in another way


# Load the workflow definitions from the JSON file
with open("workflow_definitions.json") as f:
    workflow_definitions = json.load(f)


# Function to get the latest image file in a directory
def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [
        f for f in files if f.lower().endswith(("png", "jpg", "jpeg", "gif"))
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


# Function to get the latest video file in a directory
def get_latest_video(folder):
    files = os.listdir(folder)
    video_files = [f for f in files if f.lower().endswith(("mp4", "mov"))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video


# Function to run a workflow and return the path to the output video
def run_workflow(workflow_name, *args):
    print("inside run workflow with args: " + str(args))

    # Load the workflow JSON
    workflow_json = (
        "./workflows/" + workflow_definitions[workflow_name]["filename"] + ".json"
    )
    with open(workflow_json, "r", encoding="utf-8") as f:
        workflow = json.load(f)

        # Prepare the input parameters
        params = workflow_definitions[workflow_name]["parameters"]
        for i, path in enumerate(params.values()):
            # Split the path into keys
            keys = path.strip("[]").split("][")
            keys = [key.strip('"') for key in keys]

            # Use the keys to access the corresponding value in the workflow dictionary
            sub_dict = workflow
            for key in keys[:-1]:
                sub_dict = sub_dict[key]
            sub_dict[keys[-1]] = args[i]

        current_datetime = datetime.now().strftime("%Y-%m-%d")

        output_directory = OUT_DIR

        previous_video = get_latest_video(output_directory)

        print(f"Previous video: {previous_video}")

        # Start the workflow on the server
        start_queue(workflow)

        # Wait for a new video to be created in the output directory
        while True:
            latest_video = get_latest_video(output_directory)
            if latest_video != previous_video:
                print(f"New video created: {latest_video}")
                return latest_video
            time.sleep(1)

    return ["output_video1.mp4", "output_video2.mp4"]


# Function to create a wrapper function for a workflow
def run_workflow_with_name(workflow_name):
    def wrapper(*args):
        return run_workflow(workflow_name, *args)

    return wrapper


# Function to update the preview GIF for a workflow
def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["filename"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None


# Function to create the input interface for a workflow
def create_tab_interface(workflow_name):
    components = []
    for param in workflow_definitions[workflow_name]["parameters"]:
        label = workflow_definitions[workflow_name]["labels"].get(param, param)
        if param == "text_1" or param == "text_2":
            components.append(gr.Textbox(label=label))
        elif param == "images":
            components.append(gr.Files(label=label))
        elif param == "image_path":
            components.append(gr.Textbox(label=label))
        elif param == "video_file":
            components.append(gr.File(label=label))
        elif param == "video_path":
            components.append(gr.Textbox(label=label))

    return components


# Create the Gradio interface
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
                                run_button = gr.Button("Run Workflow")
                                info = gr.Markdown(
                                    workflow_definitions[workflow_name].get("info", "")
                                )  # Add the info box here
                            with gr.Column():
                                components = create_tab_interface(workflow_name)
                            with gr.Column():
                                output_player = gr.Video(
                                    label="Output Video", autoplay=True
                                )
                                with gr.Row():
                                    mark_bad = gr.Button("Mark as Bad")
                                    mark_good = gr.Button("Mark as Good")
                                    upscale_button = gr.Button("Upscale")
                        run_button.click(
                            fn=run_workflow_with_name(workflow_name),
                            inputs=components,
                            outputs=[output_player],
                        )
    demo.launch(favicon_path="favicon.png")
