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
from lora_maker import generate_lora
import asyncio

with open("config.json") as f:
    config = json.load(f)

URL = config["COMFY_URL"]
OUT_DIR = config["COMFY_ROOT"] + "output/WorkFlower/"
LORA_DIR = config["COMFY_ROOT"] + "models/loras/"


def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    try:
        requests.post(URL, data=data)
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")


with open("workflow_definitions.json") as f:
    workflow_definitions = json.load(f)


def get_lora_filenames(directory):
    # Get all files in the directory
    files = os.listdir(directory)

    # Filter out any directories, leaving only files
    filenames = [
        file for file in files if os.path.isfile(os.path.join(directory, file))
    ]

    return filenames


# Replace with the actual path to the Loras.
loras = get_lora_filenames(LORA_DIR)


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


async def wait_for_new_video(previous_video, output_directory):
    while True:
        latest_video = get_latest_video(output_directory)
        if latest_video != previous_video:
            print(f"New video created: {latest_video}")
            return latest_video
        await asyncio.sleep(1)


def run_workflow(workflow_name, **kwargs):
    # Print the input arguments for debugging
    print("inside run workflow with kwargs: " + str(kwargs))
    print("workflow_definitions: " + str(workflow_definitions[workflow_name]))

    # Construct the path to the workflow JSON file
    workflow_json = (
        "./workflows/" + workflow_definitions[workflow_name]["filename"] + ".json"
    )

    # Open the workflow JSON file
    with open(workflow_json, "r", encoding="utf-8") as f:
        # Load the JSON data into the workflow variable
        workflow = json.load(f)

        # Get the parameters for the current workflow
        params = workflow_definitions[workflow_name]["parameters"]

        # get the human readable labels associated with those parameters
        param_labels = workflow_definitions[workflow_name]["labels"]

        # Initialize sub_dict to None
        sub_dict = None

        # Iterate over the values in the parameters dictionary
        for key, value in kwargs.items():
            # Print the current key and value for debugging
            print("######################\n\n")
            print(f"key: {key}, value: {value}")

            # Print the current state of params and sub_dict for debugging
            print("params:", params)

            # look inside the parameters for the matching kwargs key
            if key in params:
                # Get the path this value needs to be written to
                path = params.get(key)
                print(f"path: {path}")

                # If a path was found
                if path:
                    # Split the path into keys, removing brackets and splitting on ']['
                    keys = path.strip("[]").split("][")

                    # Remove any leading or trailing quotes from the keys
                    keys = [key.strip('"') for key in keys]

                    # Start with the full workflow dictionary
                    sub_dict = workflow

                    # Traverse the dictionary using the keys, stopping before the last key
                    for key in keys[:-1]:
                        sub_dict = sub_dict[key]

                    # Update the value at the last part of the path
                    print(f"updating: {sub_dict[keys[-1]]}\nto {value}")
                    sub_dict[keys[-1]] = value

        if "loras" in params:
            # now we know loras is there, get the path it should be modifying:
            path = params.get("loras")
            print(f"\n\nLora parameter detected!\nLora path: {path}")

            lora_accessors = path.strip("[]").split("][")
            lora_accessors = [key.strip('"') for key in lora_accessors]
            print(f"lora_accessors: {lora_accessors}")

            # reset the sub_dict to include all parameters again
            sub_dict = workflow

            for key in lora_accessors[:-1]:
                sub_dict = sub_dict[key]

            print(f"sub_dict: {sub_dict}")  # Debugging step 1

            if "class_type" in sub_dict:  # Debugging step 2
                print(f"class_type: {sub_dict['class_type']}")  # Debugging step 3

            if sub_dict.get("class_type") == "CR LoRA Stack":
                # for each lora in the loras array
                for lora in loras:
                    print("\n\n")
                    # set switch to the value of the checkbox (but make it 'on' or 'off' instead of True or False)
                    lora_switch = "On" if kwargs.get(f"Lora_Switch_{lora}") else "Off"
                    # get the lora name
                    lora_name = kwargs.get(f"Lora_Name_{lora}")
                    # set the model_weight to the value of the slider
                    lora_weight = kwargs.get(f"Lora_Weight_{lora}")
                    print(
                        f"lora_switch: {lora_switch}, lora_name: {lora_name}, lora_weight: {lora_weight}"
                    )
                    # set the lora details
                    switch_key = f"switch_{loras.index(lora) + 1}"
                    print(f"Switch Key: {switch_key}")
                    print(f"Before: {sub_dict['inputs'].get(switch_key)}")
                    sub_dict["inputs"][switch_key] = lora_switch
                    print(f"After: {sub_dict['inputs'].get(switch_key)}")

                    name_key = f"lora_name_{loras.index(lora) + 1}"
                    print(f"Name Key: {name_key}")
                    print(f"Before: {sub_dict['inputs'].get(name_key)}")
                    sub_dict["inputs"][name_key] = lora_name
                    print(f"After: {sub_dict['inputs'].get(name_key)}")

                    model_weight_key = f"model_weight_{loras.index(lora) + 1}"
                    # print(f"\nModel Weight Key: {model_weight_key}")
                    # print(f"Before: {sub_dict['inputs'].get(model_weight_key)}")
                    sub_dict["inputs"][model_weight_key] = lora_weight
                    # print(f"After: {sub_dict['inputs'].get(model_weight_key)}")

                    clip_weight_key = f"clip_weight_{loras.index(lora) + 1}"
                    # print(f"\nClip Weight Key: {clip_weight_key}")
                    # print(f"Before: {sub_dict['inputs'].get(clip_weight_key)}")
                    sub_dict["inputs"][clip_weight_key] = lora_weight
                    # print(f"After: {sub_dict['inputs'].get(clip_weight_key)}")
        if "image_path" in params:
            # find out where to write the path to eventually
            json_path = params.get("image_path")
            print(f"Image path detected in params! Path: {json_path}")

            image_path_accessors = json_path.strip("[]").split("][")
            image_path_accessors = [key.strip('"') for key in image_path_accessors]
            print(f"image_path_accessors: {image_path_accessors}")

            # reset the sub_dict to include all parameters again
            sub_dict = workflow

            for key in image_path_accessors[:-1]:
                sub_dict = sub_dict[key]

            print(f"sub_dict: {sub_dict}")  # Debugging step 1

            print(kwargs)
            if kwargs.get("Images Path Type") == "Nilor Collection Name":
                print(
                    f"Resolving online collection: {kwargs.get('Collection Name or Directory Path')}"
                )
                path = resolve_online_collection(
                    kwargs.get("Collection Name or Directory Path"),
                    int(kwargs.get("Max Images")),
                    kwargs.get("Shuffle Images"),
                )
                image_count = count_images(path)
                print(f"Detected {image_count} images in the collection.")
                sub_dict["directory"] = path
            else:
                print(
                    f"Loading images from local directory: {kwargs.get('Collection Name or Directory Path')}"
                )
                path = kwargs.get("Collection Name or Directory Path")
                image_count = count_images(path)
                print(f"Detected {image_count} images in the collection.")
                path = reorganise_local_files(
                    path,
                    int(kwargs.get("Max Images")),
                    kwargs.get("Shuffle Images"),
                )
                sub_dict["directory"] = path

        try:
            output_directory = OUT_DIR
            previous_video = get_latest_video(output_directory)
            print(f"Previous video: {previous_video}")
            print(f"!!!!!!!!!\nSubmitting workflow:\n{workflow}\n!!!!!!!!!")
            start_queue(workflow)
            asyncio.run(wait_for_new_video(previous_video, output_directory))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return None


def run_workflow_with_name(workflow_name, component_names):
    def wrapper(*args):
        param_labels = workflow_definitions[workflow_name]["labels"]

        # Create a reverse dictionary that maps labels to parameter names
        labels_to_param = {v: k for k, v in param_labels.items()}

        # Initialize an empty dictionary for kwargs
        kwargs = {}

        # Iterate over the arguments and their corresponding component names
        for i, arg in enumerate(args):
            # Get the label of the i-th component
            label = str(component_names[i].label)

            # Get the parameter name that corresponds to this label,
            # or use the label itself if it's not in the dictionary
            param_name = labels_to_param.get(label, label)

            # Add this argument to kwargs with the parameter name as the key
            kwargs[param_name] = arg

        print(f"kwargs in wrapper: {kwargs}")
        return run_workflow(workflow_name, **kwargs)

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
                gr.Slider(
                    label="Max Images",
                    minimum=2,
                    maximum=4,
                    value=4,
                    step=1,
                )
            )
            components.append(gr.Checkbox(label="Shuffle Images", value=False))
        elif param == "video_file":
            components.append(gr.File(label=label))
        elif param == "video_path":
            components.append(gr.Textbox(label=label))
        elif param == "bool_1":
            components.append(gr.Checkbox(label="Circular?"))
        elif param == "loras":
            with gr.Accordion(label="Lora Models", open=False):
                for lora in loras:
                    components.append(
                        gr.Checkbox(
                            label=f"Lora_Switch_{lora}",
                            value=False,
                        )
                    )
                    components.append(
                        gr.Textbox(
                            value=lora,
                            label=f"Lora_Name_{lora}",
                            visible=False,
                        )
                    )
                    components.append(
                        gr.Slider(
                            label=f"Lora_Weight_{lora}",
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.01,
                        )
                    )

    return components


with gr.Blocks(title="WorkFlower") as demo:
    with gr.Row():
        with gr.Column():
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem(label="About"):
                    with gr.Row():
                        gr.Markdown(
                            "WorkFlower is a tool for creating and running workflows. "
                            "Select a workflow from the tabs above and fill in the parameters. "
                            "Click 'Run Workflow' to start the workflow. "
                            "The output video will be displayed below."
                        )
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
                                    mark_bad = gr.Button("Mark as Bad", visible=False)
                                    mark_good = gr.Button("Mark as Good", visible=False)
                                    upscale_button = gr.Button("Upscale", visible=False)

                        run_button.click(
                            fn=run_workflow_with_name(workflow_name, components),
                            inputs=components,
                            outputs=[output_player],
                        )
                with gr.TabItem(label="LORA Maker"):
                    with gr.Row():
                        gr.Markdown(
                            "Input name of Nilor Collection to generate a LORA from the images in the collection."
                        )
                        with gr.Column():
                            lora_collection = gr.Textbox(label="Collection Name")
                            generate_lora_button = gr.Button("Generate LORA")
                        output_lora = gr.File(label="Output LORA")

                    generate_lora_button.click(
                        fn=generate_lora, inputs=lora_collection, outputs=output_lora
                    )
    demo.launch(favicon_path="favicon.png")
