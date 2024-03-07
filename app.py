import json
import os
import gradio as gr
import json
from pathlib import Path

# Load workflow definitions from JSON files
workflow_definitions = {
    "Promptable Motion": {
        "parameters": ["text_1", "text_2"],
        "filename": "promptable_motion.json"
    },
    "Steerable Motion": {
        "parameters": ["images", "image_path"],
        "filename": "steerable_motion.json"
    }
    # Add more workflow definitions here
}

def run_workflow(*args):

    print(args)

    selected_tab = args[-1]  # The label of the currently selected tab is the last argument
    # Your code here
    # Load the workflow JSON
    workflow_json = "workflows/" + workflow_definitions[workflow_name]["filename"]
    with open(workflow_json, "r") as f:
        workflow = json.load(f)

        print(workflow)

    # Prepare the input parameters
    params = {param: kwargs[param] for param in workflow_definitions[workflow_name]["parameters"]}

    # Run the workflow with the provided parameters
    # ... (Your workflow execution code goes here)

    # Return a placeholder output for demonstration
    return ["output_video1.mp4", "output_video2.mp4"]

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

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            tabs = gr.Tabs()  # Assign the Tabs component to a variable
            with tabs:
                tab_components = []
                for workflow_name in workflow_definitions.keys():
                    with gr.TabItem(label=workflow_name):
                        with gr.Row():
                            preview_gif = gr.Image(label="Preview GIF")
                            with gr.Column():
                                components = create_tab_interface(workflow_name)
                                tab_components.extend(components)
                        run_button = gr.Button("Run Workflow")
                        
        with gr.Column():
            
            with gr.Row():
                output_player = gr.Video(label="Output Video")
            with gr.Row():
                mark_bad = gr.Button("Mark as Bad")
                mark_good = gr.Button("Mark as Good")
                upscale_button = gr.Button("Upscale")
                
            run_button.click(fn=run_workflow, inputs=[*tab_components], outputs=[output_player])

    demo.launch()