import json
import requests
import os
import time
from datetime import datetime
import gradio as gr
import json
from pathlib import Path

URL = "http://127.0.0.1:8188/prompt"
OUT_DIR = "D:/SD/StabilityMatrix/Packages/ComfyUI-git/output/WorkFlower/"
output_player = None

def start_queue(prompt_workflow):
    p = {"prompt" : prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)

# Load workflow definitions from JSON files
workflow_definitions = {
    "Debug Video Gen": {
        "parameters": ["text_1"],
        "filename": "debug-video-gen"
    },
    "Promptable Motion": {
        "parameters": ["text_1", "text_2"],
        "filename": "promptable-motion"
    },
    "Steerable Motion": {
        "parameters": ["images", "image_path"],
        "filename": "steerable-motion"
    }
    # Add more workflow definitions here
}

def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

def get_latest_video(folder):
    files = os.listdir(folder)
    video_files = [f for f in files if f.lower().endswith(('mp4', 'mov'))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video


def run_workflow(workflow_name, *args):
    print(args)

    # Your code here
    # Load the workflow JSON
    workflow_json = "./workflows/" + workflow_definitions[workflow_name]["filename"] + ".json"
    with open(workflow_json, "r", encoding='utf-8') as f:
        workflow = json.load(f)

        current_datetime = datetime.now().strftime("%Y-%m-%d")

        # output_directory = OUT_DIR + current_datetime + "_/01P/"
        output_directory = OUT_DIR

        # print(output_directory)

        # previous_image = get_latest_image(output_directory)
        previous_video = get_latest_video(output_directory)

        start_queue(workflow)

        # while True:
        #     latest_image = get_latest_image(output_directory)
        #     if latest_image != previous_image:
        #         return latest_image
        #     time.sleep(1)

        while True:
            latest_video = get_latest_video(output_directory)
            if latest_video != previous_video:
                return latest_video
            time.sleep(1)

    # Prepare the input parameters
    params = {param: args[i] for i, param in enumerate(workflow_definitions[workflow_name]["parameters"])}

    # Run the workflow with the provided parameters
    # ... (Your workflow execution code goes here)

    # Return a placeholder output for demonstration
    return ["output_video1.mp4", "output_video2.mp4"]

def run_workflow_with_name(workflow_name):
    def wrapper(*args):
        return run_workflow(workflow_name, *args)
    return wrapper

def update_gif(workflow_name):
    # Load the GIF based on the selected workflow
    workflow_json = workflow_definitions[workflow_name]["filename"]
    gif_path = Path(f"gifs/{workflow_json.replace('.json', '.gif')}")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None

def create_tab_interface(workflow_name):
    components = []
    for param in workflow_definitions[workflow_name]["parameters"]:
        if param == "text_1":
            components.append(gr.Textbox(label="Text 1"))
        elif param == "text_2":
            components.append(gr.Textbox(label="Text 2"))
        elif param == "images":
            components.append(gr.Files(label="Images"))
        elif param == "image_path":
            components.append(gr.Textbox(label="Image Path"))
        elif param == "video_file":
            components.append(gr.File(label="Video File"))
        elif param == "video_path":
            components.append(gr.Textbox(label="Video Path"))

    return components

def create_output_player():
    global output_player
    output_player = gr.Video(label="Output Video")

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            tabs = gr.Tabs()  # Assign the Tabs component to a variable
            with tabs:
                for workflow_name in workflow_definitions.keys():
                    with gr.TabItem(label=workflow_name):
                        with gr.Row():
                            with gr.Column():
                                preview_gif = gr.Image(label="Preview GIF", value=f"./gifs/" + workflow_definitions[workflow_name]["filename"] + ".gif")
                                run_button = gr.Button("Run Workflow")
                            with gr.Column():
                                components = create_tab_interface(workflow_name)
                            run_button.click(fn=run_workflow_with_name(workflow_name), inputs=components, outputs=output_player)
        with gr.Column():
            create_output_player()
            with gr.Row():
                mark_bad = gr.Button("Mark as Bad")
                mark_good = gr.Button("Mark as Good")
                upscale_button = gr.Button("Upscale")

    demo.launch()