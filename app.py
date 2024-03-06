import gradio as gr
import json
from pathlib import Path

# Load workflow definitions from JSON files
workflow_definitions = {
    "Promptable Motion": {
        "parameters": {
            "text_1"
        },
    },
        "Steerable Motion": {
        "parameters": {
            "images",
            "image_path",
        },
    }
    # Add more workflow definitions here
}

def run_workflow(workflow_json, parameters):
    # Load the workflow JSON
    with open(workflow_json, "r") as f:
        workflow = json.load(f)

    # Run the workflow with the provided parameters
    # ... (Your workflow execution code goes here)

    # Return a placeholder output for demonstration
    return ["output_video1.mp4", "output_video2.mp4"]


def update_parameter_visibility(workflow_json, parameter_name):
    # Get the parameters for the selected workflow
    workflow_parameters = workflow_definitions[workflow_json]["parameters"]

    print(workflow_parameters)

    # Check if the parameter is in the workflow parameters
    if parameter_name in workflow_parameters:
        return True  # visible
    else:
        return False  # not visible


def update_gif(workflow_json):
    # Load the GIF based on the selected workflow
    gif_path = Path(f"gifs/{workflow_json.replace('.json', '.gif')}")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            workflow_dropdown = gr.Dropdown(
                choices=list(workflow_definitions.keys()),
                label="Choose Workflow",
                value=list(workflow_definitions.keys())[0]
            )

            with gr.Row():
                gr.Markdown("Workflow Parameters")


            with gr.Row():
                text_1 = gr.Textbox(label="Text 1", visible=update_parameter_visibility(workflow_dropdown.value, "text_1"))
                text_2 = gr.Textbox(label="Text 2", visible=update_parameter_visibility(workflow_dropdown.value, "text_2"))

            with gr.Row():
                image_uploads = [gr.Image(label=f"Image {i+1}", type="filepath") for i in range(4)]

            with gr.Row():
                img_path = gr.Textbox(label="Image Directory", visible=update_parameter_visibility(workflow_dropdown.value, "img_path"))

            with gr.Row():
                 video_file = gr.File(label="Video File", visible=update_parameter_visibility(workflow_dropdown.value, "video_file"))
                 video_path = gr.Textbox(label="Video Directory", visible=update_parameter_visibility(workflow_dropdown.value, "video_path"))
        
        with gr.Column():
            preview_gif = gr.Image(label="Preview GIF", value=update_gif(workflow_dropdown.value), interactive=False)

            with gr.Row():
                output_player = gr.Video(label="Output Video")

            with gr.Row():
                mark_bad = gr.Button("Mark as Bad")
                mark_good = gr.Button("Mark as Good")
                upscale_button = gr.Button("Upscale")

    run_button = gr.Button("Run Workflow")
    run_button.click(
        fn=run_workflow,
        inputs=[workflow_dropdown] ,
        outputs=[output_player]
    )

    workflow_dropdown.change(
        #fn=update_parameter_visibility,
        #inputs=workflow_dropdown,
        
    )

    workflow_dropdown.change(
        fn=update_gif,
        inputs=workflow_dropdown,
        outputs=preview_gif
    )



demo.launch()