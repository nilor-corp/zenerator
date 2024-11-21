import json
import requests
import os
#import glob
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
import threading

import websocket
import uuid
#import urllib.request
#import urllib.parse

with open("config.json") as f:
    config = json.load(f)

COMFY_IP = config["COMFY_IP"]
COMFY_PORTS = config["COMFY_PORTS"]
QUEUE_URLS = []

for port in COMFY_PORTS:
    QUEUE_URLS.append(f"http://{COMFY_IP}:{port}")

selected_port_url = QUEUE_URLS[0]

print(QUEUE_URLS)

OUT_DIR = os.path.abspath(config["COMFY_ROOT"] + "output/WorkFlower/")
LORA_DIR = os.path.abspath(config["COMFY_ROOT"] + "models/loras/")
INPUTS_DIR = os.path.abspath("./inputs/")

output_type = ""

def select_correct_port(selector):
    print(f"Selected Port URL: {selector}")
    global selected_port_url 
    selected_port_url = f"http://{COMFY_IP}:{selector}"
    print(f"Changed Port URL to: {selected_port_url}")

queue = []

prompt = {}
status = {}

previous_content = ""

#region WEBSOCKET
ws = None
client_id = str(uuid.uuid4())

def connect_to_websocket(client_id):
    ws = websocket.WebSocket()
    try:
        #TODO: make this work for multiple ports
        ws.connect(f"ws://{config['COMFY_IP']}:{config['COMFY_PORTS'][0]}/ws?clientId={client_id}")
        print("Connected to WebSocket!")
    except ConnectionResetError as e:
        print(f"Connection was reset: {e}")
        #reconnect(client_id, 20)
        return None
    return ws

def reconnect(client_id, max_retries=5):
    retries = 0
    while retries < max_retries:
        print(f"Attempting to reconnect ({retries + 1}/{max_retries})...")
        ws = connect_to_websocket(client_id)
        if ws:
            return ws
        retries += 1
        time.sleep(2)  # Wait before retrying
    print("Max retries reached. Could not reconnect.")
    return None

# def queue_prompt(prompt):
#     p = {"prompt": prompt, "client_id": client_id}
#     data = json.dumps(p).encode('utf-8')
#     req = urllib.request.Request(f"{selected_port_url}/prompt", data=data)
#     return json.loads(urllib.request.urlopen(req).read())

def send_heartbeat(ws):
    while ws.connected:
        try:
            ws.ping()
            time_as_string = datetime.now().strftime("%H:%M:%S")
            print(f"Heartbeat at {time_as_string}")
            time.sleep(1)  # Send a ping every 30 seconds
        except Exception as e:
            print(f"Heartbeat failed: {e}")
            break

check_current_progress_running = False
current_progress_data = {}
def check_current_progress(ws):
    global check_current_progress_running, current_progress_data

    check_current_progress_running = True
    #prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    progress_time = time.time()
    #print(f"Time: {progress_time}")

    while check_current_progress_running:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            #print(f"Message: {message}")
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                   check_current_progress_running = False
                   break #Execution is done
            elif message['type'] == 'status':
                data = message['data']
                queue_remaining = data['status']['exec_info']['queue_remaining']
                print("Queue remaining: " + str(queue_remaining))
            elif message['type'] == 'execution_start':
                executing = True
                print("Executing!")
            elif message['type'] == 'executed':
                data = message['data']
                prompt_id = data['prompt_id']
                print("Executed : " + prompt_id)
                #task: Task = self._tasks.get(prompt_id)
            elif message['type'] == 'progress':
                data = message['data']
                current_progress_data = data
                #progress_time_used = str(round((time.time() - start_time) / 60, 2))
                
                #print(f"Progress: {data}")
                #print("Progress is " + str(data['value']) + "/" + str(data['max']) + "time " + progress_time_used + " min")
                #print("Progress is " + str(data['value']) + "/" + str(data['max']))
                progress_time = time.time()
        else:
            continue #previews are binary data

    # history = get_history(prompt_id)[prompt_id]
    # for o in history['outputs']:
    #     for node_id in history['outputs']:
    #         node_output = history['outputs'][node_id]
    #         if 'images' in node_output:
    #             images_output = []
    #             for image in node_output['images']:
    #                 image_data = get_image(image['filename'], image['subfolder'], image['type'])
    #                 images_output.append(image_data)
    #         output_images[node_id] = images_output

    return output_images
#endregion

#region POST REQUESTS
def comfy_POST(endpoint, message):
    post_url = selected_port_url + "/" + endpoint
    data = json.dumps(message).encode("utf-8")
    print(f"POST {endpoint} on: {post_url}")
    try:
        post_response = requests.post(post_url, data=data)
        #post_response.raise_for_status()
        #print(f"status {post_response}")
        return post_response
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error querying the GET endpoint {endpoint}: ", e)

def post_prompt(prompt_workflow):
    message = {"prompt": prompt_workflow}
    return comfy_POST("prompt", message)

def post_interrupt():
    global current_progress_data, check_current_progress_running
    current_progress_data = {}

    message = ""
    return comfy_POST("interrupt", message)

def post_history_clear():
    message = {"clear": True}
    return comfy_POST("history", message)

def post_history_delete(prompt_id):
    message = {"delete": prompt_id}
    return comfy_POST("history", message)
#endregion

#region GET REQUESTS
def comfy_GET(endpoint):
    get_url = selected_port_url + "/" + endpoint
    #print(f"GET {endpoint} on: {get_url}\n")
    try:
        return requests.get(get_url).json()
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error querying the POST endpoint {endpoint}: ", e)
        
def get_queue():
    global queue, queue_running, queue_pending, queue_failed

    queue = comfy_GET("queue")
    if (queue is None):
        print("/queue GET response is empty")
        return [[], [], []]
    
    queue_running = queue.get("queue_running", [])
    #print(f"queue_running: {len(queue_running)}")
    
    queue_pending = queue.get("queue_pending", [])
    #print(f"queue_pending: {len(queue_pending)}")

    queue_failed = queue.get("queue_failed", [])
    #print(f"queue_failed: {len(queue_failed)}")

    return [queue_running, queue_pending, queue_failed]

def get_running_prompt_id():
    [queue_running, queue_pending, queue_failed] = get_queue()

    if (len(queue_running) > 0):
        prompt_id = queue_running[0][1]
        print(f"current running prompt id: {prompt_id}")
        return prompt_id
    else:
        return None
    
def get_status():
    global prompt, status

    prompt = comfy_GET("prompt")
    if (prompt is None):
        print("/prompt GET response is empty")
        return "N/A"
    
    status = prompt.get("status", "N/A")
    #print(f"status: {status}")

    return status

def get_history():
    global history

    history = comfy_GET("history")
    if (history is None):
        print("/history GET response is empty")
        return {}

    #print(f"history: {len(history)}")

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
    
    #print(f"devices: {devices}")

    #for device in devices:
        #print(f"device['name']: {device.get("name")}")
        #print(f"device['torch_vram_free']: {device.get("torch_vram_free")}")
        #print(f"device['torch_vram_total']: {device.get("torch_vram_total")}")

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


#region Content Getters
def get_all_images(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []
    
    files = os.listdir(folder)
    image_files = [
        f for f in files if f.lower().endswith(("png", "jpg", "jpeg", "gif"))
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return image_files

def get_latest_image(folder):
    image_files = get_all_images(folder)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

def get_latest_image_with_prefix(folder, prefix):
    image_files = get_all_images(folder)
    image_files = [
        f for f in image_files if f.lower().startswith(prefix)
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

# def count_images(directory):
#     extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"]
#     image_count = sum(
#         len(glob.glob(os.path.join(directory, ext))) for ext in extensions
#     )
#     return image_count

def get_all_videos(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []
    
    files = os.listdir(folder)
    video_files = [
        f for f in files if f.lower().endswith(("mp4", "mov"))
    ]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return video_files

def get_latest_video(folder):
    video_files = get_all_videos(folder)
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video

def get_latest_video_with_prefix(folder, prefix):
    video_files = get_all_videos(folder)
    video_files = [
        f for f in video_files if f.lower().startswith(prefix)
    ]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video
#endregion

#region Info Checkers
def check_for_new_content():
    global latest_content, previous_content
    #print(f"Checking for new content in: {OUT_DIR}\n")

    latest_content = ""
    if output_type == "image":
        latest_content = get_latest_image(OUT_DIR)
    else:
        latest_content = get_latest_video(OUT_DIR)

    if latest_content != previous_content:
        print(f"New content found: {latest_content}")
        previous_content = latest_content

    #output_filepath_component = gr.Markdown(f"{latest_content}")

    return gr.update(value=latest_content)

previous_vram_used = -1.0
check_vram_running = False
def check_vram(progress=gr.Progress()):
    global check_vram_running, previous_vram_used

    check_vram_running = True

    while check_vram_running:
        [system_stats, devices] = get_system_stats()
        
        # if system_stats is None or len(devices) == 0:
        #     progress(progress=0.0)
            # vram_used = 0.0

        vram_used = 0.0
        if (len(devices) > 0):
            vram_free = devices[0].get("vram_free")
            vram_total = devices[0].get("vram_total")
            if (vram_total > 0):
                vram_used = (vram_total - vram_free) / vram_total

        #print(f"vram_used: {vram_used}")
        
        if (vram_used != previous_vram_used):
            progress(progress=vram_used)
            previous_vram_used = vram_used

        time.sleep(1.0)

previous_queue_info = [-1, -1, -1, -1, -1, -1]
current_queue_info = [-1, -1, -1, -1, -1, -1]
check_queue_running = False
def check_queue(progress=gr.Progress()):
    global check_queue_running, previous_queue_info, current_queue_info

    check_queue_running = True

    while check_queue_running:
        [queue_running, queue_pending, queue_failed] = get_queue()
        queue_history = get_history()
        
        queue_finished = len(queue_history)
        queue_total = queue_finished + len(queue_running) + len(queue_pending)

        #print(f"queue_progress: {queue_finished} out of {queue_total}")
        
        current_queue_info = [queue_finished, queue_total, queue_failed, queue_history, len(queue_pending), len(queue_running)]
        if queue_total < 1:
            progress(progress=0.0)
            #return gr.update(visible=False)
        elif (current_queue_info != previous_queue_info):
            previous_queue_info = current_queue_info
            progress(progress=(queue_finished, queue_total), unit="generations")
            #return gr.update(visible=True)

        time.sleep(1.0)

check_progress_running = False
previous_progress_data = {}
def check_progress(progress=gr.Progress()):
    global check_progress_running, current_progress_data, previous_progress_data, ws
    
    check_progress_running = True

    while check_progress_running:
        try:
            if current_progress_data != previous_progress_data:
                previous_progress_data = current_progress_data

            progress_tuple = (current_progress_data.get("value", 0), current_progress_data.get("max",0))
            prompt_id = current_progress_data.get("prompt_id", "N/A")
            if prompt_id != "N/A":
                prompt_id_short = prompt_id[:8]
                progress(progress=progress_tuple, unit="steps", desc=prompt_id_short)
            else:
                progress(progress=0.0)
            #return gr.update(visible=True)
        except:
            progress(progress=0.0)
            #return gr.update(visible=False)

        if not ws or not ws.connected:
            break
        
        time.sleep(1.0)

def check_gen_progress_visibility():
    global current_progress_data, current_queue_info
    try: 
        prompt_id = current_progress_data.get("prompt_id", None)
        current_step = current_progress_data.get("value", None)

        queue_running = current_queue_info[5]

        #print(f"Prompt ID: {prompt_id}, Current Step: {current_step}")
        visibility = (current_step != None) and (queue_running > 0)
    except:
        visibility = False
    return gr.update(visible=visibility)

def check_interrupt_visibility():
    global current_progress_data, current_queue_info
    try: 
        prompt_id = current_progress_data.get("prompt_id", None)
        current_step = current_progress_data.get("value", None)

        queue_running = current_queue_info[5]

        #print(f"Prompt ID: {prompt_id}, Current Step: {current_step}, Queue Running: {queue_running}")
        visibility = (current_step != None) and (queue_running > 0)
    except:
        visibility = False
    return gr.update(visible=visibility)
#endregion

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
            print(f"!!!!!!!!!\nSubmitting workflow:\n{workflow}\n!!!!!!!!!")

            post_prompt(workflow)

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
        #print(f"Reveal option input: {selected_option}")
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
    gr.Markdown(f"##### {input_type.capitalize()} Input", elem_classes="group-label")    
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

    #possible_inputs = select_dynamic_input_option(selected_option.value, choices)

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
    
def make_visible():
    return gr.update(visible=True)

def watch_input(component, default_value, elem_id):
    print(f"Equals Default Value? {component == default_value}")
    if component != default_value:
        # Return HTML to change the background color when value changes
        # html = f"<style>#{elem_id}  {{ background: #30435d; }}"
        html = ""
        return gr.update(value=html, visible=True), gr.update(visible=True)
    else:
        # Return HTML to reset background color when value matches default
        # html = f"<style>#{elem_id}  {{ background: var(--input-background-fill); }}"
        html = ""
        return gr.update(value=html, visible=False), gr.update(visible=False)

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
                gr.Markdown(f"##### {input_label}", elem_classes="group-label")    
                
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
                    with gr.Row(equal_height=True):
                        # Checkbox component which enables Group
                        component_constructor = component_map.get(input_type)
                        group_toggle = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100)
                        
                        # Compact Reset button with reduced width, initially hidden
                        reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)
                        # Trigger the reset function when the button is clicked
                        reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=group_toggle, queue=False, show_progress="hidden")

                        # Trigger the reset check when the value of the input changes
                        html_output = gr.HTML(visible=False)
                        group_toggle.change(fn=watch_input, inputs=[group_toggle, gr.State(input_value), gr.State(input_key)], outputs=[html_output, reset_button], queue=False, show_progress="hidden")
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
                group_toggle.change(fn=toggle_group, inputs=group_toggle, outputs=input_group, queue=False, show_progress="hidden")
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
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component_constructor = component_map.get(input_type)
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, minimum=input_minimum, maximum=input_maximum, step=input_step, interactive=input_interactive, scale=100)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)
                    # Trigger the reset function when the button is clicked
                    reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=component, queue=False, show_progress="hidden")

                # Trigger the reset check when the value of the input changes
                html_output = gr.HTML(visible=False)
                component.change(fn=watch_input, inputs=[component, gr.State(input_value), gr.State(input_key)], outputs=[html_output, reset_button], queue=False, show_progress="hidden")

                components.append(component)
                components_dict[input_key] = input_details

                # print(f"Component Constructor: {component_constructor}")
            else:
                if input_type == "path":
                    input_value = os.path.abspath(input_value)
                    
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component_constructor = component_map.get(input_type)
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)
                    # Trigger the reset function when the button is clicked
                    reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=component, queue=False, show_progress="hidden")

                # Trigger the reset check when the value of the input changes
                html_output = gr.HTML(visible=False)
                component.change(fn=watch_input, inputs=[component, gr.State(input_value), gr.State(input_key)], outputs=[html_output, reset_button], queue=False, show_progress="hidden")

                components.append(component)
                components_dict[input_key] = input_details

                # print(f"Component Constructor: {component_constructor}")
        else:
            print(f"Whoa! Unsupported input type: {input_type}")

    return [components, components_dict]
    #return components

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


def load_demo():
    global ws, tick_timer, check_vram_running, previous_vram_used, check_queue_running, previous_queue_info, current_queue_info, check_progress_running, previous_progress_data
    print("Loading the demo!!!")

    tick_timer = gr.Timer(value=1.0)
    ws = connect_to_websocket(client_id)

    if ws:
        try:
            # Start threading after connecting
            print(f"WebSocket connection attempt")
            threading.Thread(target=send_heartbeat, args=(ws,), daemon=True).start()
            threading.Thread(target=check_current_progress, args=(ws,), daemon=True).start()
        except websocket.WebSocketConnectionClosedException as e:
            print(f"WebSocket connection closed: {e}")
            ws = reconnect(ws, client_id)
        except Exception as e:
            print(f"An error occurred: {e}")

    check_vram_running = False
    previous_vram_used = -1.0

    check_queue_running = False
    previous_queue_info = [-1, -1, -1, -1, -1, -1]
    current_queue_info = [-1, -1, -1, -1, -1, -1]

    check_progress_running = False
    previous_progress_data = {}

def unload_demo():
    global ws, tick_timer, check_vram_running, previous_vram_used, check_queue_running, previous_queue_info, current_queue_info, check_progress_running, previous_progress_data
    print("Unloading the demo...")

    tick_timer.active = False
    ws.close()

    check_vram_running = False
    previous_vram_used = -1.0

    check_queue_running = False
    previous_queue_info = [-1, -1, -1, -1, -1, -1]
    current_queue_info = [-1, -1, -1, -1, -1, -1]

    check_progress_running = False
    previous_progress_data = {}

    time.sleep(2.0)

custom_css = """
.group-label {
    padding: .25rem;
}

#workflow-info {
    background-image: linear-gradient(120deg, var(--neutral-800) 0%, var(--neutral-900) 70%, var(--primary-800) 100%);
}

html {
    overflow-y: scroll; /* Always show vertical scrollbar */
}
"""    

with gr.Blocks(title="WorkFlower", theme=gr.themes.Ocean(font=gr.themes.GoogleFont("DM Sans")), css=custom_css) as demo:
    tick_timer = gr.Timer(value=1.0)
    demo.load(fn=load_demo)

    demo.unload(fn=unload_demo)

    with gr.Row():
        with gr.Column(scale=5):

            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem(label="About"):
                    with gr.Row():
                        gr.Markdown(
                            "WorkFlower is a tool for creating and running workflows.\n\n"
                            "Select a workflow from the tabs above and fill in the parameters.\n\n"
                            "Click 'Run Workflow' to start the workflow.  ",
                            #"The output video will be displayed below."
                            line_breaks=True,
                        )
                for workflow_name in workflow_definitions.keys():
                    workflow_filename = workflow_definitions[workflow_name]["filename"]

                    # make a tab for each workflow
                    with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):

                        with gr.Row():
                            # main input construction
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row(equal_height=True):
                                        comfy_url_and_port_selector = gr.Dropdown(label="ComfyUI Port", choices=COMFY_PORTS, value=COMFY_PORTS[0], interactive=True, scale=1)
                                        print(f"Default ComfyUI Port: {comfy_url_and_port_selector.value}")
                                        comfy_url_and_port_selector.change(select_correct_port, inputs=[comfy_url_and_port_selector])    
                                        run_button = gr.Button("Run Workflow", variant="primary", scale=2)
                                    with gr.Accordion("Workflow Info", open=False, elem_id="workflow-info"):
                                        info = gr.Markdown(workflow_definitions[workflow_name].get("description", ""))

                                # also make a dictionary with the components' data
                                components, component_dict = create_tab_interface(workflow_name)
                        
                            if (selected_port_url is not None) and (components is not None) and (component_dict is not None):
                                run_button.click(
                                    fn=run_workflow_with_name(workflow_filename, components, component_dict),
                                    inputs=components,
                                    #outputs=[gen_component],
                                    trigger_mode="multiple",
                                    #show_progress="full"
                                )

                            output_type = workflow_definitions[workflow_name]["outputs"].get("type", "")
                            #output_prefix = workflow_definitions[workflow_name]["inputs"]["output-specifications"]["inputs"]["filename-prefix"].get("value", "")
                        
        with gr.Column(scale=4):
            # TODO: is it possible to preview only an output that was produced by this workflow tab? otherwise this should probably exist outside of the workflow tab
            gr.Markdown("### Output Preview")
            with gr.Group():
                if output_type == "image":
                    output_player = gr.Image(show_label=False, interactive=False)
                else:
                    output_player = gr.Video(show_label=False, autoplay=True, loop=True, interactive=False)
                #output_filepath_component = gr.Markdown("N/A")
                
                tick_timer.tick(
                    fn=check_for_new_content,
                    outputs=[output_player],
                    show_progress="hidden"
                )
                
            with gr.Group(visible=False) as gen_progress_group:
                gen_component = gr.Textbox(label="Generation Progress", interactive=False, visible=True)
            
                tick_timer.tick(
                    fn=check_progress,
                    outputs=gen_component, 
                    show_progress="full"
                )

            tick_timer.tick(
                fn=check_gen_progress_visibility, 
                outputs=gen_progress_group,
                show_progress="hidden"
            )

            with gr.Group(visible=False) as queue_progress_group:
                queue_info_component = gr.Textbox(label="Queue Info", interactive=False, visible=True)

                tick_timer.tick(
                    fn=check_queue, 
                    outputs=queue_info_component, 
                    show_progress="full"
                )
                
            tick_timer.tick(
                fn=check_interrupt_visibility, 
                outputs=queue_progress_group,
                show_progress="hidden"
            )

            with gr.Group():
                vram_usage_component = gr.Textbox(label="VRAM Usage", interactive=False, visible=True)
                
                tick_timer.tick(
                    fn=check_vram, 
                    outputs=[vram_usage_component], 
                    show_progress="full"
                )

            with gr.Group() as interrupt_group:
                interrupt_button = gr.Button("Interrupt", visible=False, variant="stop")
                interrupt_button.click(fn=post_interrupt)
            
            tick_timer.tick(
                fn=check_interrupt_visibility, 
                outputs=interrupt_group, 
                show_progress="hidden"
            )

    demo.launch(allowed_paths=[".."], favicon_path="favicon.png")