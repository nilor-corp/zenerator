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
import socket

with open("config.json") as f:
    config = json.load(f)

COMFY_URL = config["COMFY_URL"]
QUEUE_URL = config["COMFY_URL"] + "/prompt"
OUT_DIR = config["COMFY_ROOT"] + "output/WorkFlower/"
LORA_DIR = config["COMFY_ROOT"] + "models/loras/"

output_type = ""


def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    try:
        requests.post(QUEUE_URL, data=data)
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


async def wait_for_new_content(previous_content, output_directory):
    while True:
        latest_content = ""
        if output_type == "video":
            latest_content = get_latest_video(output_directory)
        elif output_type == "image":
            latest_content = get_latest_image(output_directory)
        if latest_content != previous_content:
            print(f"New content created: {latest_content}")
            return latest_content
        await asyncio.sleep(1)


def run_workflow(workflow_name, **kwargs):
    # Print the input arguments for debugging
    print("inside run workflow with kwargs: " + str(kwargs))
    print("workflow_definitions: " + str(workflow_definitions[workflow_name]))

    # Construct the path to the workflow JSON file
    workflow_json = (
        "./workflows/" + workflow_name + ".json"
    )

    # Open the workflow JSON file
    with open(workflow_json, "r", encoding="utf-8") as f:
        # Load the JSON data into the workflow variable
        workflow = json.load(f)

        # Get the parameters for the current workflow
        params = workflow_definitions[workflow_name]["inputs"]

        # get the human readable labels associated with those parameters
        # param_labels = workflow_definitions[workflow_name]["labels"]

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

        # Process cases where there should be filenames of images submitted, rather than paths
        image_filenames = []

        if "image_filename_1" in params:
            image_filenames.append("image_filename_1")
        if "image_filename_2" in params:
            image_filenames.append("image_filename_2")

        for image_filename in image_filenames:
            print(f"image_filename: {image_filename}")
            if image_filename is None:
                return
            else:
                print(params[image_filename])

                # get path for the final image filename to go to
                json_path = params.get(image_filename)
                image_filename_accessors = json_path.strip("[]").split("][")
                image_filename_accessors = [
                    key.strip('"') for key in image_filename_accessors
                ]
                print(f"image_filename_accessors: {image_filename_accessors}")

                # reset the sub_dict to include all parameters again
                sub_dict = workflow

                for key in image_filename_accessors[:-1]:
                    sub_dict = sub_dict[key]

                print(f"sub_dict: {sub_dict}")  # Debugging step

                # take a gradio input and POST it to the api input folder
                img_path = kwargs.get(image_filename)
                post_url = f"{COMFY_URL}/upload/image"
                data = {
                    "overwrite": "false",
                    "subfolder": "",
                }

                print(f"Posting image to {post_url}")
                print(f"Data: {data}")

                try:
                    with open(img_path, "rb") as img_file:
                        files = {"image": img_file}
                        response = requests.post(post_url, files=files, data=data)
                except ConnectionResetError:
                    print(
                        "Connection was reset. The remote host may have forcibly closed the connection."
                    )
                except socket.error as e:
                    print(f"Socket error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

                # get the POST response, which contains the actual filename that comfy can see
                try:
                    data = response.json()
                    image_filename_from_POST = data["name"]
                except json.JSONDecodeError:
                    print("Invalid JSON response:", response.text)

                # update the workflow json with the filename
                sub_dict["image"] = image_filename_from_POST

        try:
            output_directory = OUT_DIR

            previous_content = ""

            if output_type == "video":
                previous_content = get_latest_video(output_directory)
                print(f"Previous video: {previous_content}")
            elif output_type == "image":
                previous_content = get_latest_image(output_directory)
                print(f"Previous image: {previous_content}")

            # print(f"!!!!!!!!!\nSubmitting workflow:\n{workflow}\n!!!!!!!!!")
            start_queue(workflow)

            asyncio.run(wait_for_new_content(previous_content, output_directory))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return None


def run_workflow_with_name(workflow_name, raw_components, component_info_dict):
    def wrapper(*args):

        # Initialize an empty dictionary for kwargs
        kwargs = {}

        # attach the component label to the arg
        for component, arg in zip(raw_components, args):

            # access the component_info_dict using component.elem_id and add a value field = arg
            component_info_dict[component.elem_id]["value"] = arg

        # print(f"\nComponent Info Dict: {component_info_dict}\n")

        return run_workflow(workflow_name, **component_info_dict)

    return wrapper




def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["name"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None


def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")
    # create the actual gradio components
    components = []
    # create a dictionary to store the components and their data
    component_data_dict = {workflow_name:workflow_definitions[workflow_name]["inputs"]}

    # print(f"\n{workflow_name}")
    
    for input_key in workflow_definitions[workflow_name]["inputs"]:
        input_details = workflow_definitions[workflow_name]["inputs"][input_key]
        input_type = input_details["type"]
        input_label = input_details["label"]
        input_node_id = input_details["node-id"]
        # Now you can use input_type, input_label, and input_node_id as needed
        # print(f"\nInput Key: {input_key}")
        # print(f"Type: {input_type}, Label: {input_label}, Node-ID: {input_node_id}")
        
        # Define a mapping of input types to Gradio components
        component_map = {
            "text": gr.Textbox,
            "image": gr.Image,
            "video": gr.File,
            "bool": gr.Checkbox,
            "float": gr.Number,
            "int": lambda label: gr.Number(label=label, precision=0)  # Special case for int to round
        }

        # Use the mapping to create components based on input_type
        component_constructor = component_map.get(input_type)
        if component_constructor:
            # add the component to the list of components
            components.append(component_constructor(label=input_label, elem_id=input_key))
            
        else:
            print(f"Unsupported input type: {input_type}")

    return components, component_data_dict


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
                    # make a tab for each workflow
                    with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    preview_gif = gr.Image(
                                        label="Preview GIF",
                                        value=update_gif(workflow_name),
                                    )
                                info = gr.Markdown(
                                    workflow_definitions[workflow_name].get("description", "")
                                )
                            # main input construction
                            with gr.Column():
                                run_button = gr.Button("Run Workflow")
                                # also make a dictionary with the components' data
                                components, component_dict = create_tab_interface(workflow_name)
                            # output player construction
                            with gr.Column():
                                output_type = workflow_definitions[workflow_name]["outputs"].get(
                                    "type", ""
                                )
                                if output_type == "video":
                                    output_player = gr.Video(
                                        label="Output Video", autoplay=True
                                    )
                                elif output_type == "image":
                                    output_player = gr.Image(label="Output Image")

                        # investigate trigger_mode=multiple for run_button.click event

                        run_button.click(
                            fn=run_workflow_with_name(workflow_name, components, component_dict[workflow_name]),
                            inputs=components,
                            outputs=[output_player],
                        )





    #             with gr.TabItem(label="LORA Maker"):
    #                 with gr.Row():
    #                     gr.Markdown(
    #                         "Input name of Nilor Collection to generate a LORA from the images in the collection."
    #                     )
    #                     with gr.Column():
    #                         lora_collection = gr.Textbox(label="Collection Name")
    #                         generate_lora_button = gr.Button("Generate LORA")
    #                     output_lora = gr.File(label="Output LORA")

    #                 generate_lora_button.click(
    #                     fn=generate_lora, inputs=lora_collection, outputs=output_lora
    #                 )
    demo.launch(favicon_path="favicon.png")
