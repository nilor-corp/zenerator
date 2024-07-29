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
    # print("workflow_definitions: " + str(workflow_definitions[workflow_name]))


    # Construct the path to the workflow JSON file
    workflow_json = (
        "./workflows/" + workflow_name + ".json"
    )

    # Open the workflow JSON file
    with open(workflow_json, "r", encoding="utf-8") as file:
        # Load the JSON data
        workflow = json.load(file)
    
        # Iterate through changes requested via kwargs
        for change_request in kwargs.values():
            # Extract the node path and the new value from the change request
            node_path = change_request['node-id']
            new_value = change_request['value']
    
            # Log the intended change for debugging
            print(f"Intending to change {node_path} to {new_value}")
    
            # Process the node path into a list of keys
            path_keys = node_path.strip("[]").split("][")
            path_keys = [key.strip('"') for key in path_keys]
    
            # Navigate through the workflow data to the last key
            current_section = workflow
            for key in path_keys[:-1]:  # Exclude the last key for now
                current_section = current_section[key]
    
            # Update the value at the final key
            final_key = path_keys[-1]
            print(f"Updating {current_section[final_key]} to {new_value}")
            current_section[final_key] = new_value












        # if "loras" in params:
        #     # now we know loras is there, get the path it should be modifying:
        #     path = params.get("loras")
        #     print(f"\n\nLora parameter detected!\nLora path: {path}")

        #     lora_accessors = path.strip("[]").split("][")
        #     lora_accessors = [key.strip('"') for key in lora_accessors]
        #     print(f"lora_accessors: {lora_accessors}")

        #     # reset the sub_dict to include all parameters again
        #     sub_dict = workflow

        #     for key in lora_accessors[:-1]:
        #         sub_dict = sub_dict[key]

        #     print(f"sub_dict: {sub_dict}")  # Debugging step 1

        #     if "class_type" in sub_dict:  # Debugging step 2
        #         print(f"class_type: {sub_dict['class_type']}")  # Debugging step 3

        #     if sub_dict.get("class_type") == "CR LoRA Stack":
        #         # for each lora in the loras array
        #         for lora in loras:
        #             print("\n\n")
        #             # set switch to the value of the checkbox (but make it 'on' or 'off' instead of True or False)
        #             lora_switch = "On" if kwargs.get(f"Lora_Switch_{lora}") else "Off"
        #             # get the lora name
        #             lora_name = kwargs.get(f"Lora_Name_{lora}")
        #             # set the model_weight to the value of the slider
        #             lora_weight = kwargs.get(f"Lora_Weight_{lora}")
        #             print(
        #                 f"lora_switch: {lora_switch}, lora_name: {lora_name}, lora_weight: {lora_weight}"
        #             )
        #             # set the lora details
        #             switch_key = f"switch_{loras.index(lora) + 1}"
        #             print(f"Switch Key: {switch_key}")
        #             print(f"Before: {sub_dict['inputs'].get(switch_key)}")
        #             sub_dict["inputs"][switch_key] = lora_switch
        #             print(f"After: {sub_dict['inputs'].get(switch_key)}")

        #             name_key = f"lora_name_{loras.index(lora) + 1}"
        #             print(f"Name Key: {name_key}")
        #             print(f"Before: {sub_dict['inputs'].get(name_key)}")
        #             sub_dict["inputs"][name_key] = lora_name
        #             print(f"After: {sub_dict['inputs'].get(name_key)}")

        #             model_weight_key = f"model_weight_{loras.index(lora) + 1}"
        #             # print(f"\nModel Weight Key: {model_weight_key}")
        #             # print(f"Before: {sub_dict['inputs'].get(model_weight_key)}")
        #             sub_dict["inputs"][model_weight_key] = lora_weight
        #             # print(f"After: {sub_dict['inputs'].get(model_weight_key)}")

        #             clip_weight_key = f"clip_weight_{loras.index(lora) + 1}"
        #             # print(f"\nClip Weight Key: {clip_weight_key}")
        #             # print(f"Before: {sub_dict['inputs'].get(clip_weight_key)}")
        #             sub_dict["inputs"][clip_weight_key] = lora_weight
        #             # print(f"After: {sub_dict['inputs'].get(clip_weight_key)}")
        # if "image_path" in params:
        #     # find out where to write the path to eventually
        #     json_path = params.get("image_path")
        #     print(f"Image path detected in params! Path: {json_path}")

        #     image_path_accessors = json_path.strip("[]").split("][")
        #     image_path_accessors = [key.strip('"') for key in image_path_accessors]
        #     print(f"image_path_accessors: {image_path_accessors}")

        #     # reset the sub_dict to include all parameters again
        #     sub_dict = workflow

        #     for key in image_path_accessors[:-1]:
        #         sub_dict = sub_dict[key]

        #     print(f"sub_dict: {sub_dict}")  # Debugging step 1

        #     print(kwargs)
        #     if kwargs.get("Images Path Type") == "Nilor Collection Name":
        #         print(
        #             f"Resolving online collection: {kwargs.get('Collection Name or Directory Path')}"
        #         )
        #         path = resolve_online_collection(
        #             kwargs.get("Collection Name or Directory Path"),
        #             int(kwargs.get("Max Images")),
        #             kwargs.get("Shuffle Images"),
        #         )
        #         image_count = count_images(path)
        #         print(f"Detected {image_count} images in the collection.")
        #         sub_dict["directory"] = path
        #     else:
        #         print(
        #             f"Loading images from local directory: {kwargs.get('Collection Name or Directory Path')}"
        #         )
        #         path = kwargs.get("Collection Name or Directory Path")
        #         image_count = count_images(path)
        #         print(f"Detected {image_count} images in the collection.")
        #         path = reorganise_local_files(
        #             path,
        #             int(kwargs.get("Max Images")),
        #             kwargs.get("Shuffle Images"),
        #         )
        #         sub_dict["directory"] = path

        # # Process cases where there should be filenames of images submitted, rather than paths
        # image_filenames = []

        # if "image_filename_1" in params:
        #     image_filenames.append("image_filename_1")
        # if "image_filename_2" in params:
        #     image_filenames.append("image_filename_2")

        # for image_filename in image_filenames:
        #     print(f"image_filename: {image_filename}")
        #     if image_filename is None:
        #         return
        #     else:
        #         print(params[image_filename])

        #         # get path for the final image filename to go to
        #         json_path = params.get(image_filename)
        #         image_filename_accessors = json_path.strip("[]").split("][")
        #         image_filename_accessors = [
        #             key.strip('"') for key in image_filename_accessors
        #         ]
        #         print(f"image_filename_accessors: {image_filename_accessors}")

        #         # reset the sub_dict to include all parameters again
        #         sub_dict = workflow

        #         for key in image_filename_accessors[:-1]:
        #             sub_dict = sub_dict[key]

        #         print(f"sub_dict: {sub_dict}")  # Debugging step

        #         # take a gradio input and POST it to the api input folder
        #         img_path = kwargs.get(image_filename)
        #         post_url = f"{COMFY_URL}/upload/image"
        #         data = {
        #             "overwrite": "false",
        #             "subfolder": "",
        #         }

        #         print(f"Posting image to {post_url}")
        #         print(f"Data: {data}")

        #         try:
        #             with open(img_path, "rb") as img_file:
        #                 files = {"image": img_file}
        #                 response = requests.post(post_url, files=files, data=data)
        #         except ConnectionResetError:
        #             print(
        #                 "Connection was reset. The remote host may have forcibly closed the connection."
        #             )
        #         except socket.error as e:
        #             print(f"Socket error: {e}")
        #         except Exception as e:
        #             print(f"An unexpected error occurred: {e}")

        #         # get the POST response, which contains the actual filename that comfy can see
        #         try:
        #             data = response.json()
        #             image_filename_from_POST = data["name"]
        #         except json.JSONDecodeError:
        #             print("Invalid JSON response:", response.text)

        #         # update the workflow json with the filename
        #         sub_dict["image"] = image_filename_from_POST

        try:
            output_directory = OUT_DIR

            previous_content = ""

            if output_type == "video":
                previous_content = get_latest_video(output_directory)
                print(f"Previous video: {previous_content}")
            elif output_type == "image":
                previous_content = get_latest_image(output_directory)
                print(f"Previous image: {previous_content}")

            print(f"!!!!!!!!!\nSubmitting workflow:\n{workflow}\n!!!!!!!!!")
            start_queue(workflow)

            asyncio.run(wait_for_new_content(previous_content, output_directory))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return None


def run_workflow_with_name(workflow_name, raw_components, component_info_dict):
    
    for component in raw_components:
        print(f"Component: {component.label}")

    def wrapper(*args):

        # match the component to the arg
        for component, arg in zip(raw_components, args):
            # access the component_info_dict using component.elem_id and add a value field = arg
            component_info_dict[component.elem_id]["value"] = arg

        return run_workflow(workflow_name, **component_info_dict)

    return wrapper


def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["name"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None



def create_dynamic_input(selected_input, label, input_key):
    print(f"Selected input: {selected_input}")
    val = selected_input
    if isinstance(val, str):
        if val == "filepath" or val == "nilor collection":
            return gr.Textbox(label=label, elem_id=input_key)
        elif val == "upload":
            return gr.Gallery(label=label, elem_id=input_key)
        else:
            print(f"Unsupported input type: {selected_input}")
            return None

def render_dynamic_component(choice, label, input_key):
    print(f"\nAttempting to render dynamic component with choice: {choice}")

    @gr.render(inputs=choice)
    def render_input(current_choice):

        if isinstance(current_choice, str):
            print(f"\nCurrent GUI choice: {current_choice}")
            dynamic_input = create_dynamic_input(current_choice, label, input_key)
            # print(f"Dynamic input: {dynamic_input}")
            return dynamic_input
    
    return render_input(choice)


def update_dynamic_component(choice, label, input_key, components):

    @gr.render(inputs= [choice, label, input_key, components])
    def update_component(choice, label, input_key, components):
        if isinstance(choice, gr.Radio):
            print(f"\nChoice: {choice.value}, Label: {label}, Input Key: {input_key}")
            dynamic_component = render_dynamic_component(choice.value, label, input_key)
            print(f"\nDynamic component: {dynamic_component}")
            if dynamic_component:
                components.append(dynamic_component)
                print(f"Dynamically added component: {dynamic_component} with label: {dynamic_component.label} and elem_id: {dynamic_component.elem_id}")  
        else:
            print(f"Not a Radio: {choice}")

    update_component(choice, label, input_key, components)

    return components

def create_radio():
    return gr.Radio(choices=["filepath", "nilor collection", "upload"], label="Select Input Type", value="filepath")


# Ensure all elements in self.inputs are valid Gradio components
def filter_valid_components(components):
    valid_components = []
    for component in components:
        if hasattr(component, '_id'):
            valid_components.append(component)
    return valid_components


def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")
    components = []
    component_data_dict = {workflow_name: workflow_definitions[workflow_name]["inputs"]}
    print(f"\nWORKFLOW: {workflow_name}")


    for input_key in workflow_definitions[workflow_name]["inputs"]:
        input_details = workflow_definitions[workflow_name]["inputs"][input_key]
        input_type = input_details["type"]
        input_label = input_details["label"]
        input_node_id = input_details["node-id"]
        


        # Define a mapping of input types to Gradio components
        component_map = {
            "text": gr.Textbox,
            "image": create_radio,
            "video": gr.File,
            "bool": gr.Checkbox,
            "float": gr.Number,
            "int": gr.Number  # Special case for int to round
        }



        # print(f"Creating: {component_constructor} for input type: {input_type}")

        if input_type in component_map:
            if input_type == "image":
                print("!!!!!!!!!!!!!!!!!!!!!!!\nMaking Radio")
                # Make the radio button
                radio_component = gr.Radio(choices=["filepath", "nilor collection", "upload"], label="Select Input Type", value="filepath")
                print(f"Radio Component: {radio_component}\nPerforming initial update")
                
                components = update_dynamic_component(radio_component, input_label, input_key, components)
                
                print(f"Assigning change function to radio component with inputs: {radio_component}, {input_label}, {input_key}, {components}")


                # radio_component.input(
                #     fn=update_dynamic_component,
                #     inputs=[radio_component, input_label, input_key, components],
                #     outputs=components
                # )
            else:
                # Use the mapping to create components based on input_type
                component_constructor = component_map.get(input_type)

                components.append(component_constructor(label=input_label, elem_id=input_key))
        else:
            print(f"Whoa! Unsupported input type: {input_type}")


    print("filtering...")
    filter_valid_components(components)
    

    print(f"@@ Components: {components}")
    for component in components:
        print(f"@@ Component: {component.label}")

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
                            trigger_mode="multiple",
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
