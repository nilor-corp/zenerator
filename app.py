import gradio as gr
import json

# Load workflow definitions from JSON files
workflow_definitions = {
    "workflow1.json": {
        "parameters": {
            "text": "str",
            "image_dir": "str",
            "video_dir": "str",
            "options_text": "str"
        }
    }
    # Add more workflow definitions here.
}

def run_workflow(workflow_json, parameters):
    # Load the workflow JSON
    with open(workflow_json, "r") as f:
        workflow = json.load(f)

    # Run the workflow with the provided parameters
    # ... (Your workflow execution code goes here)

    # Return a placeholder output for demonstration
    return "Placeholder output"

def update_parameters(workflow_json):
    # Get the parameter definitions for the selected workflow
    parameter_definitions = workflow_definitions.get(workflow_json, {}).get("parameters", {})

    # Create a dictionary to store the parameter components
    parameter_components = {}

    # Create input components for each parameter
    for param_name, param_type in parameter_definitions.items():
        if param_type == "str":
            parameter_components[param_name] = gr.Textbox(label=param_name)
        elif param_type == "int":
            parameter_components[param_name] = gr.Number(label=param_name, type="number")
        # Add more parameter types as needed

    return parameter_components

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        workflow_dropdown = gr.Dropdown(
            choices=list(workflow_definitions.keys()),
            label="Choose Workflow",
            value=list(workflow_definitions.keys())[0]
        )

    with gr.Row():
        parameter_group = gr.Group(update_parameters(workflow_dropdown.value))

    with gr.Row():
        input_gif = gr.Image(label="Preview GIF")
        output_text = gr.Textbox(label="Output")

    run_button = gr.Button("Run Workflow")
    run_button.click(
        fn=run_workflow,
        inputs=[workflow_dropdown, parameter_group],
        outputs=[output_text]
    )

demo.launch()