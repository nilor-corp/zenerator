import json
import requests
import os
import glob
import time
from datetime import datetime
import gradio as gr
from pathlib import Path
from image_downloader import resolve_online_collection
from image_downloader import organise_local_files
from image_downloader import copy_uploaded_files_to_local_dir
from lora_maker import generate_lora
from tqdm import tqdm
import asyncio
import socket

with open("config.json") as f:
    config = json.load(f)

COMFY_URL = config["COMFY_URL"]

QUEUE_URLS = []

for port in config["COMFY_PORTS"]:
    QUEUE_URLS.append(config["COMFY_URL"] + port)

selected_port_url = QUEUE_URLS[0]

print(QUEUE_URLS)

OUT_DIR = os.path.abspath(config["COMFY_ROOT"] + "output/WorkFlower/")
LORA_DIR = os.path.abspath(config["COMFY_ROOT"] + "models/loras/")
INPUTS_DIR = os.path.abspath("./inputs/")

output_type = ""

def select_correct_port(selector):
    print(f"Selected Port URL: {selector}")
    global selected_port_url 
    selected_port_url = selector
    print(f"Changed Port URL to: {selected_port_url}")

queue = []
queue_running = []
queue_pending = []
queue_failed = []

history = {}

prompt = {}
status = {}

system_stats = {}
devices = []

previous_content = ""

#region POST REQUESTS
def comfy_POST(endpoint, message):
    post_url = selected_port_url + "/" + endpoint
    data = json.dumps(message).encode("utf-8")
    print(f"POST {endpoint} on: {post_url}")
    try:
        return requests.post(post_url, data=data)
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error polling the GET {endpoint}: ", e)

def post_prompt(prompt_workflow):
    message = {"prompt": prompt_workflow}
    comfy_POST("prompt", message)

def post_history_clear():
    p = {"clear": True}
    data = json.dumps(p).encode("utf-8")
#endregion

#region GET REQUESTS
def comfy_GET(endpoint):
    get_url = selected_port_url + "/" + endpoint
    print(f"GET {endpoint} on: {get_url}\n")
    try:
        return requests.get(get_url).json()
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error polling the POST {endpoint}: ", e)
        
def get_queue():
    global queue, queue_running, queue_pending, queue_failed

    queue = comfy_GET("queue")
    if (queue is None):
        print("/queue GET response is empty")
        return [[], [], []]
    
    queue_running = queue.get("queue_running", [])
    print(f"queue_running: {len(queue_running)}")
    
    queue_pending = queue.get("queue_pending", [])
    print(f"queue_pending: {len(queue_pending)}")

    queue_failed = queue.get("queue_failed", [])
    print(f"queue_failed: {len(queue_failed)}")

    return [queue_running, queue_pending, queue_failed]
    
def get_status():
    global prompt, status

    prompt = comfy_GET("prompt")
    if (prompt is None):
        print("/prompt GET response is empty")
        return "N/A"
    
    status = prompt.get("status", "N/A")
    print(f"status: {status}")

    return status

def get_history():
    global history

    history = comfy_GET("history")
    if (history is None):
        print("/history GET response is empty")
        return {}

    print(f"history: {len(history)}")

    return history

def get_system_stats():
    global system_stats, devices

    system_stats = comfy_GET("system_stats")
    if (system_stats is None):
        print("/system_stats GET response is empty")
        return [[], []]
    
    devices = system_stats.get("devices")
    if (devices is None):
        return [system_stats, []]
    
    print(f"devices: {devices}")

    for device in devices:
        #print(f"device: {device}")
        print(f"device['name']: {device.get("name")}")
        print(f"device['vram_free']: {device.get("vram_free")}")
        print(f"device['vram_total']: {device.get("vram_total")}")

    return [system_stats, devices]
#endregion
    

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


def check_for_new_content():
    global latest_content, previous_content

    print(f"Checking for new content in: {OUT_DIR}\n")

    latest_content = ""

    latest_content = get_latest_video(OUT_DIR)
    output_player = gr.Video(label=f"Output Video", show_label=False, autoplay=True, loop=True, value=latest_content)
    if output_type == "image":
        latest_content = get_latest_image(OUT_DIR)
        output_player = gr.Image(label="Output Image", show_label=False, value=latest_content)

    if latest_content != previous_content:
        print(f"New content created: {latest_content}")
        previous_content = latest_content

    output_filepath_component = gr.Markdown(f"{latest_content}")
    
    return [output_player, output_filepath_component]


def run_workflow(workflow_name, progress, **kwargs):
    global previous_content

    # Print the input arguments for debugging
    print("inside run workflow with kwargs: " + str(kwargs))
    # print("workflow_definitions: " + str(workflow_definitions[workflow_name]))


    # Construct the path to the workflow JSON file
    workflow_json = (
        "./workflows/" + workflow_name
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

        try:
            previous_content = ""

            if output_type == "video":
                previous_content = get_latest_video(OUT_DIR)
                print(f"Previous video: {previous_content}")
            elif output_type == "images":
                previous_content = get_latest_image(OUT_DIR)
                print(f"Previous image: {previous_content}")

            print(f"!!!!!!!!!\nSubmitting workflow:\n{workflow}\n!!!!!!!!!")
            post_prompt(workflow)

            # asyncio.run(check_for_new_content(output_directory))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return None


def run_workflow_with_name(workflow_name, raw_components, component_info_dict, progress=gr.Progress(track_tqdm=True)):
    for component in raw_components:
        print(f"Component: {component.label}")

    def wrapper(*args):
        # match the component to the arg
        for component, arg in zip(raw_components, args):
            # access the component_info_dict using component.elem_id and add a value field = arg
            component_info_dict[component.elem_id]["value"] = arg

        return run_workflow(workflow_name, progress, **component_info_dict, )

    return wrapper


def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["name"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None


def select_dynamic_input_option(selected_option, choices):
    print(f"Selected option: {selected_option}")
    # print(f"Choices: {choices}")
    updaters = [gr.update(visible=False) for _ in choices]
    
    # Make the corresponding input visible based on the selected option
    if selected_option in choices:
        selected_index = choices.index(selected_option)
        updaters[selected_index] = gr.update(visible=True)

    return updaters

def process_dynamic_input(selected_option, possible_options, input_type, *option_values):
    print("\nProcessing dynamic input")
    print(f"Selected Option: {selected_option}")
    print(f"Possible Options: {possible_options}")
    print(f"Option Values: {option_values}")

    # Get the selected option
    selected_index = possible_options.index(selected_option)
    selected_value = option_values[selected_index]
    print(f"Selected Value: {selected_value}")

    # process the selected value based on the selected option
    if selected_option == "filepath":
        return selected_value
    elif selected_option == "nilor collection":
        return resolve_online_collection(selected_value, None, False)
    elif selected_option == "upload":
        return copy_uploaded_files_to_local_dir(selected_value, input_type, None, False)
    else:
        return None


def create_dynamic_input(input_type, choices, tooltips, text_label, identifier):
    gr.Markdown(f"##### {input_type.capitalize()} Input")    
    with gr.Group():            
        selected_option = gr.Radio(choices, label=text_label, value=choices[0])
        print(f"Choices: {choices}")
        if input_type == "images":
            possible_inputs = [
                gr.Textbox(label=choices[0], show_label=False, visible=True, info=tooltips[0]),
                gr.Textbox(label=choices[1], show_label=False, visible=False, info=tooltips[1]),
                gr.Gallery(label=choices[2], show_label=False, visible=False)
            ]
        elif input_type == "video":
            possible_inputs = [
                gr.Textbox(label=choices[0], show_label=False, visible=True, info=tooltips[0]),
                gr.File(label=choices[1], show_label=False, visible=False, file_count="single", type="filepath", file_types=["video"])
            ]


        output = gr.Markdown(elem_id=identifier)
        # output = os.path.abspath(output)

    # modify visibility of inputs based on selected_option
    selected_option.change(select_dynamic_input_option, inputs=[selected_option, gr.State(choices)], outputs=possible_inputs)


    print(f"Possible Inputs: {possible_inputs}")
    for input_box in possible_inputs:
        if isinstance(input_box, gr.Textbox):
            input_box.submit(process_dynamic_input, inputs=[selected_option, gr.State(choices), gr.State(input_type)] + possible_inputs, outputs=output)
        elif isinstance(input_box, gr.Gallery) or isinstance(input_box, gr.File):
            input_box.upload(process_dynamic_input, inputs=[selected_option, gr.State(choices), gr.State(input_type)] + possible_inputs, outputs=output)
    return selected_option, possible_inputs, output

# Ensure all elements in self.inputs are valid Gradio components
def filter_valid_components(components):
    valid_components = []
    for component in components:
        if hasattr(component, '_id'):
            valid_components.append(component)
    return valid_components

def toggle_group(checkbox_value):
    # If checkbox is selected, the group of inputs will be visible
    if checkbox_value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def watch_input(component, default_value, elem_id):
    #print(f"Equals Default Value? {component == default_value}")
    if component != default_value:
        # Return HTML to change the background color to red when value changes
        output = f"<style>#{elem_id}  {{ background: #30435d; }}"
        return output, gr.update(visible=True)
    else:
        # Return HTML to reset background color when value matches default
        output = f"<style>#{elem_id}  {{ background: var(--input-background-fill); }}"
        return output, gr.update(visible=False)

def reset_input(default_value):
    return default_value

def process_input(input_context, input_key):
    input_details = input_context.get(input_key, None)
    input_type = input_details.get("type", None)
    input_label = input_details.get("label", None)
    input_node_id = input_details.get("node-id", None)
    input_value = input_details.get("value", None)
    input_interactive = input_details.get("interactive", True)
    input_minimum = input_details.get("minimum", None)
    input_maximum = input_details.get("maximum", None)
    input_step = input_details.get("step", 1)

    # Define a mapping of input types to Gradio components
    component_map = {
        "path": gr.Textbox,
        "string": gr.Textbox,
        "text": gr.Textbox,
        "images": None, # special case for radio selection handled below
        "video": None, # special case for video selection handled below
        "bool": gr.Checkbox,
        "float": gr.Number,
        "int": gr.Number,
        "group": None,
        "toggle-group": gr.Checkbox
    }
    
    components = []
    components_dict = {}

    with gr.Group():
        if input_type in component_map:
            if input_type == "group":
                gr.Markdown(f"##### {input_label}")    
                
                with gr.Group():
                    # Group of inputs
                    with gr.Group():
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(sub_context, group_input_key)

                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)
            elif input_type == "toggle-group":
                with gr.Group():
                    with gr.Row():
                        # Checkbox component which enables Group
                        component_constructor = component_map.get(input_type)
                        group_toggle = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100)
                        
                        # Compact Reset button with reduced width, initially hidden
                        reset_button = gr.Button("Reset", visible=False, elem_id="reset-button", scale=1, variant="stop", min_width=50)
                        # Trigger the reset function when the button is clicked
                        reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=group_toggle, queue=False)

                    # Group of inputs (initially hidden)
                    with gr.Group(visible=group_toggle.value) as input_group:
                        # Use the mapping to create components based on input_type
                        components.append(group_toggle)
                        components_dict[input_key] = input_details

                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(sub_context, group_input_key)
                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)

                # Update the group visibility based on the checkbox
                group_toggle.change(fn=toggle_group, inputs=group_toggle, outputs=input_group, queue=False)
                # Trigger the reset check when the value of the input changes
                group_toggle.change(fn=watch_input, inputs=[group_toggle, gr.State(input_value), gr.State(input_key)], outputs=[gr.HTML(), reset_button], queue=False)
            elif input_type == "images":
                # print("!!!!!!!!!!!!!!!!!!!!!!!\nMaking Radio")
                selected_option, inputs, output = create_dynamic_input(
                    input_type,
                    choices=["filepath", "nilor collection", "upload"], 
                    tooltips=["Enter the path of the directory of images and press Enter to submit", "Enter the name of the Nilor Collection and press Enter to resolve"],
                    text_label="Select Input Type", 
                    identifier=input_key
                )

                # Only append the output (Markdown element) to the components list
                components.append(output)
                components_dict[input_key] = input_details
            elif input_type == "video":
                selected_option, inputs, output = create_dynamic_input(
                    input_type,
                    choices=["filepath", "upload"], 
                    tooltips=["Enter the path of the directory of video and press Enter to submit"],
                    text_label="Select Input Type", 
                    identifier=input_key
                )

                # Only append the output (Markdown element) to the components list
                component = components.append(output)
                components_dict[input_key] = input_details
            elif input_type == "float" or input_type == "int":
                with gr.Row():
                    #gr.Markdown(f"{input_label}")
                    # Use the mapping to create components based on input_type
                    component_constructor = component_map.get(input_type)
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, minimum=input_minimum, maximum=input_maximum, step=input_step, interactive=input_interactive, scale=100)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("Reset", visible=False, elem_id="reset-button", scale=1, variant="stop", min_width=50)
                    # Trigger the reset function when the button is clicked
                    reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=component, queue=False)

                # Trigger the reset check when the value of the input changes
                component.change(fn=watch_input, inputs=[component, gr.State(input_value), gr.State(input_key)], outputs=[gr.HTML(), reset_button], queue=False)

                components.append(component)
                components_dict[input_key] = input_details

                # print(f"Component Constructor: {component_constructor}")
            else:
                if input_type == "path":
                    input_value = os.path.abspath(input_value)
                    
                with gr.Row():
                    # Use the mapping to create components based on input_type
                    component_constructor = component_map.get(input_type)
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("Reset", visible=False, elem_id="reset-button", scale=1, variant="stop", min_width=50)
                    # Trigger the reset function when the button is clicked
                    reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=component, queue=False)

                # Trigger the reset check when the value of the input changes
                component.change(fn=watch_input, inputs=[component, gr.State(input_value), gr.State(input_key)], outputs=[gr.HTML(), reset_button], queue=False)

                components.append(component)
                components_dict[input_key] = input_details

                # print(f"Component Constructor: {component_constructor}")
        else:
            print(f"Whoa! Unsupported input type: {input_type}")

    return [components, components_dict]
    #return components

timer_active = False

# start and stop timer are used for live updating the preview and progress
# no point in keeping the timer ticking if it's not currently generating
def start_timer(): 
    global timer_active
    timer_active = True
    return gr.Timer(active=True)

def stop_timer():
    global timer_active
    timer_active = False
    return gr.Timer(active=False)

def update_queue_info():
    print("TICK queue info")
    
    [queue_running, queue_pending, queue_failed] = get_queue()
    queue_running_component = gr.Markdown(f"Queue running: {len(queue_running)}")
    queue_pending_component = gr.Markdown(f"Queue pending: {len(queue_pending)}")
    queue_failed_component = gr.Markdown(f"Queue failed: {len(queue_failed)}")

    queue_history = get_history()
    queue_history_component = gr.Markdown(f"Queue history: {len(queue_history)}")

    return [queue_running_component, queue_pending_component, queue_failed_component, queue_history_component]

def update_system_stats():
    print("TICK stats")
    
    [system_stats, devices] = get_system_stats()
    
    vram_usage = "N/A"
    if (len(devices) > 0):
        vram_used = (1.0 - devices[0].get("vram_free") / devices[0].get("vram_total")) * 100.0
        vram_usage = "{:.2f}".format(vram_used) + "%"
        
    vram_usage_component = gr.Markdown(f"VRAM usage: {vram_usage}")

    return vram_usage_component

def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")

    key_context = workflow_definitions[workflow_name]["inputs"]

    components = []
    component_data_dict = {}
    
    #constants = []
    #constants_data_dict = {workflow_name: workflow_definitions[workflow_name]["constants"]}

    print(f"\nWORKFLOW: {workflow_name}")

    interactive_inputs = []
    noninteractive_inputs = []

    interactive_components = []
    noninteractive_components = []

    for input_key in key_context:
        input_details = key_context[input_key]
        input_interactive = input_details.get("interactive", True)

        if input_interactive:
            interactive_inputs.append(input_key)
        else:
            noninteractive_inputs.append(input_key)

    for input_key in interactive_inputs:
        [sub_components, sub_dict_values] = process_input(key_context, input_key)
        interactive_components.extend(sub_components)
        component_data_dict.update(sub_dict_values)

    with gr.Accordion("Constants", open=False):
        gr.Markdown("You can edit these constants in workflow_definitions.json if you know what you're doing.")
        
        for input_key in noninteractive_inputs:
            [sub_components, sub_dict_values] = process_input(key_context, input_key)
            noninteractive_components.extend(sub_components)
            component_data_dict.update(sub_dict_values)

    components.extend(interactive_components)
    components.extend(noninteractive_components)
    
    return components, component_data_dict


with gr.Blocks(title="WorkFlower") as demo:
    tick_timer = gr.Timer(value=1.0)

    with gr.Row():
        with gr.Column():
            comfy_url_and_port_selector = gr.Dropdown(label="ComfyUI Prompt URL", choices=QUEUE_URLS, value=QUEUE_URLS[0], interactive=True)
            print(f"Default Port URL: {comfy_url_and_port_selector.value}")
            comfy_url_and_port_selector.change(select_correct_port, inputs=[comfy_url_and_port_selector])
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem(label="About"):
                    with gr.Row():
                        gr.Markdown(
                            "WorkFlower is a tool for creating and running workflows. "
                            "Select a workflow from the tabs above and fill in the parameters. "
                            "Click 'Run Workflow' to start the workflow. "
                            #"The output video will be displayed below."
                        )
                for workflow_name in workflow_definitions.keys():
                    workflow_filename = workflow_definitions[workflow_name]["filename"]

                    # make a tab for each workflow
                    with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):
                        info = gr.Markdown(workflow_definitions[workflow_name].get("description", ""))

                        with gr.Row():
                            # main input construction
                            with gr.Column():
                                # also make a dictionary with the components' data
                                components, component_dict = create_tab_interface(workflow_name)
                                run_button = gr.Button("Run Workflow", variant="primary")

                            with gr.Column():
                                # TODO: Decide whether it's worth having to make a gif for each workflow tab
                                # with gr.Row():
                                #     preview_gif = gr.Image(
                                #         label="Preview GIF",
                                #         value=update_gif(workflow_name),
                                #     )

                                # TODO: is it possible to preview only an output that was produced by this workflow tab? otherwise this should probably exist outside of the workflow tab
                                gr.Markdown("### Output Preview")
                                with gr.Group():
                                    output_player = gr.Video()
                                    if output_type == "image":
                                        output_player = gr.Image()
                                    output_filepath_component = gr.Markdown("Output Filepath: N/A")
                                    tick_timer.tick(fn=check_for_new_content, outputs=[output_player, output_filepath_component])

                                gr.Markdown("### Queue Info")
                                with gr.Group():
                                    queue_running_component = gr.Markdown(f"Queue running: N/A")
                                    queue_pending_component = gr.Markdown(f"Queue pending: N/A")
                                    queue_history_component = gr.Markdown(f"Queue history: N/A")
                                    queue_failed_component = gr.Markdown(f"Queue failed: N/A")
                                tick_timer.tick(fn=update_queue_info, outputs=[queue_running_component, queue_pending_component, queue_failed_component, queue_history_component])

                                gr.Markdown("### System Stats")
                                with gr.Group():       
                                    vram_usage_component = gr.Markdown(f"VRAM Usage: N/A")
                                tick_timer.tick(fn=update_system_stats, outputs=[vram_usage_component])
                            
                                output_type = workflow_definitions[workflow_name]["outputs"].get(
                                    "type", ""
                                )

                        # TODO: investigate trigger_mode=multiple for run_button.click event

                        if (selected_port_url is not None) and (components is not None) and (component_dict is not None):
                                run_button.click(
                                fn=run_workflow_with_name(workflow_filename, components, component_dict),
                                inputs=components,
                                # TODO: Add support for real progress bar
                                #outputs=[output_player],
                                trigger_mode="multiple",
                            )


    demo.launch(allowed_paths=[".."], favicon_path="favicon.png")

