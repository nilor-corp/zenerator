import json
import requests
import os

# import glob
import time
from datetime import datetime
import gradio as gr
from pathlib import Path
from image_downloader import (
    resolve_online_collection,
    organise_local_files,
    copy_uploaded_files_to_local_dir,
    process_video_file,
)
from lora_maker import generate_lora
from tqdm import tqdm
import asyncio
import threading

import websocket
import uuid

import torch

import signal
import sys

from enum import Enum
import traceback

with open("config.json") as f:
    config = json.load(f)

COMFY_IP = config["COMFY_IP"]
COMFY_PORTS = config["COMFY_PORTS"]
QUEUE_URLS = []

ROOT_DIR = str(Path(__file__).parent.resolve())
COMFY_ROOT = str(Path(config["COMFY_ROOT"]).resolve())

OUT_DIR = str((Path(COMFY_ROOT) / "output" / "Zenerator").resolve())
MODELS_DIR = str((Path(COMFY_ROOT) / "models").resolve())
LORA_DIR = str((Path(MODELS_DIR) / "loras").resolve())
INPUTS_DIR = str((Path(ROOT_DIR) / "inputs").resolve())

allowed_paths = []
base_paths = [
    os.path.abspath(COMFY_ROOT),
    os.path.abspath(OUT_DIR),
    os.path.abspath(LORA_DIR),
    os.path.abspath(INPUTS_DIR),
]

for path in base_paths:
    allowed_paths.append(str(path))
    allowed_paths.append(str(path).replace("\\", "/"))

print(f"Allowed paths: {allowed_paths}")

TENSORRT_NODES_DIR = str(
    (Path(COMFY_ROOT) / "custom_nodes" / "ComfyUI-Upscaler-Tensorrt").resolve()
)
TENSORRT_DIR = str((Path(MODELS_DIR) / "tensorrt").resolve())
UPSCALER_DIR = str((Path(TENSORRT_DIR) / "upscaler").resolve())
UPSCALER_PATH = str((Path(UPSCALER_DIR) / "realistic.engine").resolve())
ONNX_PATH = str((Path(ROOT_DIR) / "models" / "realistic.onnx").resolve())

# Ensure required directories exist
for directory in [OUT_DIR, INPUTS_DIR, TENSORRT_DIR, UPSCALER_DIR]:
    os.makedirs(directory, exist_ok=True)

for port in COMFY_PORTS:
    QUEUE_URLS.append(f"http://{COMFY_IP}:{port}")

selected_port_url = QUEUE_URLS[0]

running = True
output_type = ""
threads = []
queue = []
prompt = {}
status = {}
previous_content = None
tick_timer = None
download_progress = {"current": 0, "total": 0, "status": ""}

job_tracking = {}
current_output_type = "image"

initial_timestamp = 0  # Time when the system started

latest_known_image = None
latest_known_video = None


class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"


def signal_handler(signum, frame):
    global running, tick_timer, threads
    print("\nShutdown signal received. Cleaning up...")
    running = False

    # Stop all monitoring loops
    check_vram_running = False
    check_queue_running = False
    check_progress_running = False

    # Deactivate timer
    if tick_timer:
        tick_timer.active = False
    tick_timer = None

    # Close WebSocket
    if ws:
        try:
            ws.close()
        except websocket.WebSocketException as e:
            print(f"Error closing websocket: {e}")
        except ConnectionError as e:
            print(f"Connection error while closing: {e}")

    # Wait for threads to finish
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=2.0)

    sys.exit(0)


def select_correct_port(selector):
    print(f"Selected Port URL: {selector}")
    global selected_port_url
    selected_port_url = f"http://{COMFY_IP}:{selector}"
    print(f"Changed Port URL to: {selected_port_url}")


# region TensorRT
def check_tensorrt_installation():
    print("Checking TensorRT installation...")

    installed = False
    if not os.path.exists(TENSORRT_DIR):
        os.makedirs(TENSORRT_DIR)

    if not os.path.exists(UPSCALER_DIR):
        os.makedirs(UPSCALER_DIR)

    if os.path.exists(UPSCALER_PATH):
        print("Realistic TensorRT engine found.")
        installed = True

    if not installed:
        print("TensorRT is not installed!")
        # exec(open(TENSORRT_NODES_DIR + "export_trt.py").read())

    return installed


# REF: https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/blob/master/export_trt.py
def install_tensorrt():
    print("Installing TensorRT...")

    engine = Engine(UPSCALER_DIR + "\\realistic.engine")

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        ONNX_PATH,
        True,
        enable_preview=True,
        input_profile=[
            {
                "input": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 1280, 1280)]
            },  # any sizes from 256x256 to 1280x1280
        ],
    )
    e = time.time()
    print(f"Time taken to build TensorRT: {(e-s)} seconds")

    return ret


if os.path.exists(TENSORRT_NODES_DIR):
    print(f"Importing TensorRT requirements.")

    sys.path.append(TENSORRT_NODES_DIR)

    # from export_trt import export_trt
    from utilities import Engine

try:
    if check_tensorrt_installation():
        print(f"TensorRT is already installed.")
    else:
        install_tensorrt()
except Exception as e:
    print(f"Error installing TensorRT: {e}")
# endregion

# region Websocket
ws = None
client_id = str(uuid.uuid4())


def connect_to_websocket(client_id):
    ws = websocket.WebSocket()
    try:
        # TODO: make this work for multiple ports
        ws.connect(
            f"ws://{config['COMFY_IP']}:{config['COMFY_PORTS'][0]}/ws?clientId={client_id}"
        )
    except ConnectionResetError as e:
        print(f"Connection was reset: {e}")
        # reconnect(client_id, 20)
        return None
    except Exception as e:
        print(f"Exception while connecting: {e}")
        # reconnect(client_id, 20)
        return None

    print("Connected to WebSocket successfully!")
    return ws


def reconnect(client_id, max_retries=5):
    retries = 0
    while retries < max_retries:
        print(f"Attempting to reconnect ({retries + 1}/{max_retries})...")
        ws = connect_to_websocket(client_id)
        if ws:
            return ws
        retries += 1
        time.sleep(1)  # Wait before retrying
    print("Max retries reached. Could not reconnect.")
    return None


# def queue_prompt(prompt):
#     p = {"prompt": prompt, "client_id": client_id}
#     data = json.dumps(p).encode('utf-8')
#     req = urllib.request.Request(f"{selected_port_url}/prompt", data=data)
#     return json.loads(urllib.request.urlopen(req).read())


def send_heartbeat(ws):
    while ws and ws.connected and running:  # Add running check
        try:
            ws.ping()
            time_as_string = datetime.now().strftime("%H:%M:%S")
            print(f"Heartbeat at {time_as_string}")
            time.sleep(1)
        except websocket.WebSocketConnectionClosedException:
            print("WebSocket connection closed, stopping heartbeat")
            break
        except ConnectionError as e:
            print(f"Connection error in heartbeat: {e}")
            break
        except Exception as e:
            print(f"Unexpected error in heartbeat: {type(e).__name__}: {e}")
            break


check_current_progress_running = False
current_progress_data = {}


def check_current_progress():
    global executing, ws, current_progress_data
    try:
        while running:
            # Poll the status endpoint for job progress
            status = get_status()
            if status != "N/A" and isinstance(status, dict):
                if "exec_info" in status:
                    current_progress_data = {
                        "value": status["exec_info"].get("value", 0),
                        "max": status["exec_info"].get("max", 0),
                        "prompt_id": status.get("prompt_id", "N/A"),
                    }
                    print(f"Updated progress data: {current_progress_data}")  # Debug
                else:
                    print(f"No progress data found in status: {status}")

            # Still use websocket for execution events
            if ws:
                try:
                    if not ws.connected:
                        print("WebSocket disconnected, waiting for reconnection...")
                        time.sleep(1)
                        continue

                    message = ws.recv()
                    if message is not None:
                        message = json.loads(message)

                        if message["type"] == "execution_start":
                            executing = True
                            print("Executing!")
                        elif message["type"] == "executed":
                            prompt_id = message["data"]["prompt_id"]
                            print("Executed: " + prompt_id)

                except websocket.WebSocketConnectionClosedException:
                    print("WebSocket connection closed normally")
                    time.sleep(1)
                except Exception as e:
                    print(f"Error in websocket loop: {str(e)}")  # Debug
                    time.sleep(1)

            time.sleep(1)  # Poll every second

    except Exception as e:
        print(f"Error in check_current_progress: {str(e)}")  # Debug


# endregion


# region POST Requests
def comfy_POST(endpoint, message):
    post_url = selected_port_url + "/" + endpoint
    data = json.dumps(message).encode("utf-8")
    print(f"POST {endpoint} on: {post_url}")
    try:
        post_response = requests.post(post_url, data=data)
        # post_response.raise_for_status()
        # print(f"status {post_response}")
        return post_response
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error querying the GET endpoint {endpoint}: ", e)


def post_prompt(workflow):
    """Submit a workflow prompt to ComfyUI"""
    prompt_data = {"prompt": workflow, "client_id": "app"}

    try:
        print(f"Attempting to post prompt to {selected_port_url}/prompt")
        print(f"Prompt data: {prompt_data}")

        response = requests.post(f"{selected_port_url}/prompt", json=prompt_data)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")

        if response.status_code != 200:
            print(f"Error posting prompt: {response.status_code} - {response.text}")
            return None

        data = response.json()
        print(f"Parsed response data: {data}")

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            print("No prompt_id in response data")
            return None

        print(f"Successfully got prompt_id: {prompt_id}")
        return prompt_id

    except Exception as e:
        print(f"Exception in post_prompt: {str(e)}")
        print(f"Stack trace: ", traceback.format_exc())
        return None


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


# endregion


# region GET Requests
def comfy_GET(endpoint):
    get_url = selected_port_url + "/" + endpoint
    # print(f"GET {endpoint} on: {get_url}\n")
    try:
        return requests.get(get_url).json()
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error querying the POST endpoint {endpoint}: ", e)


def get_queue():
    """Get the current queue status"""
    try:
        queue = comfy_GET("queue")
        queue_running = queue.get("queue_running", [])
        queue_pending = queue.get("queue_pending", [])
        queue_failed = queue.get("queue_failed", [])
        return [queue_running, queue_pending, queue_failed]
    except Exception as e:
        print(f"Error getting queue: {str(e)}")
        return [[], [], []]


def get_running_prompt_id():
    [queue_running, queue_pending, queue_failed] = get_queue()

    if len(queue_running) > 0:
        prompt_id = queue_running[0][1]
        print(f"current running prompt id: {prompt_id}")
        return prompt_id
    else:
        return None


def get_status():
    """Get the current execution status"""
    global prompt, status

    try:
        prompt = comfy_GET("prompt")
        if prompt is None:
            print("/prompt GET response is empty")
            return "N/A"

        status = prompt.get("status", "N/A")
        print(f"Prompt endpoint data: {prompt}")  # Debug
        return status

    except Exception as e:
        print(f"Error getting prompt status: {e}")
        return "N/A"


def get_history():
    global history

    history = comfy_GET("history")
    if history is None:
        print("/history GET response is empty")
        return {}

    # print(f"history: {len(history)}")

    return history


def get_system_stats():
    global system_stats, devices

    system_stats = comfy_GET("system_stats")
    if system_stats is None:
        print("/system_stats GET response is empty")
        return [[], []]

    devices = system_stats.get("devices")
    if devices is None:
        return [system_stats, []]

    # print(f"devices: {devices}")

    # for device in devices:
    # print(f"device['name']: {device.get("name")}")
    # print(f"device['torch_vram_free']: {device.get("torch_vram_free")}")
    # print(f"device['torch_vram_total']: {device.get("torch_vram_total")}")

    return [system_stats, devices]


# endregion

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


# region Content Getters
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
    image_files = get_all_images(folder)  # Already sorted in get_all_images
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


def get_latest_image_with_prefix(folder, prefix):
    image_files = get_all_images(folder)
    image_files = [f for f in image_files if f.lower().startswith(prefix)]
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
    video_files = [f for f in files if f.lower().endswith(("mp4", "mov"))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return video_files


def get_latest_video(folder):
    video_files = get_all_videos(folder)
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video


def get_latest_video_with_prefix(folder, prefix):
    video_files = get_all_videos(folder)
    video_files = [f for f in video_files if f.lower().startswith(prefix)]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video


def get_latest_content(folder, type):
    if type == "image":
        return get_latest_image(folder)
    elif type == "video":
        return get_latest_video(folder)


# endregion


# region Info Checkers
def get_content_type_from_workflow(workflow_name: str) -> ContentType:
    """Get content type from workflow definition"""
    return (
        ContentType.VIDEO
        if workflow_definitions[workflow_name]["outputs"]["type"] == "video"
        else ContentType.IMAGE
    )


def track_generation_job(job_id: str, workflow_name: str):
    """Track a new generation job"""
    content_type = get_content_type_from_workflow(workflow_name)
    job_tracking[job_id] = {
        "status": "pending",
        "type": content_type,
        "workflow_name": workflow_name,
        "timestamp": time.time(),
        "output_file": None,  # Track the specific output file
    }


def initialize_content_tracking():
    """Record the latest content files at startup"""
    global latest_known_image, latest_known_video
    print("\nInitializing content tracking...")

    latest_known_image = get_latest_content(OUT_DIR, ContentType.IMAGE.value)
    latest_known_video = get_latest_content(OUT_DIR, ContentType.VIDEO.value)

    print(f"Initial image: {latest_known_image}")
    print(f"Initial video: {latest_known_video}")


def check_for_new_content():
    """Check for new content and associate with correct job"""
    global latest_known_image, latest_known_video

    try:
        # Only log job tracking when there are NEW pending jobs
        pending_jobs_now = {
            pid for pid, job in job_tracking.items() if job["status"] == "pending"
        }
        pending_jobs_now = set(pending_jobs_now)  # Convert to set for comparison
        if not hasattr(check_for_new_content, "last_pending_jobs"):
            check_for_new_content.last_pending_jobs = set()

        if pending_jobs_now != check_for_new_content.last_pending_jobs:
            print(f"\nPending jobs changed: {len(pending_jobs_now)} jobs waiting")
            print(f"Current job tracking: {job_tracking}")
            check_for_new_content.last_pending_jobs = pending_jobs_now

        for content_type in ContentType:
            current_latest = get_latest_content(OUT_DIR, content_type.value)
            if current_latest is None:
                continue

            # Compare with our known latest file
            known_latest = (
                latest_known_image
                if content_type == ContentType.IMAGE
                else latest_known_video
            )

            if current_latest == known_latest:
                continue

            print(f"\nFound new {content_type.value}: {current_latest}")

            # Find pending jobs of this type
            pending_jobs = {
                pid: data
                for pid, data in job_tracking.items()
                if data["status"] == "pending"
                and data["type"] == content_type
                and data["output_file"] is None
            }

            if not pending_jobs:
                # Update our known latest even without a job
                if content_type == ContentType.IMAGE:
                    latest_known_image = current_latest
                else:
                    latest_known_video = current_latest
                print(
                    f"No pending jobs for new {content_type.value}, updated known latest"
                )
                continue

            # Get the oldest pending job
            oldest_job_id = min(
                pending_jobs.keys(), key=lambda pid: pending_jobs[pid]["timestamp"]
            )

            # Assign the file to the job
            job_tracking[oldest_job_id].update(
                {"status": "completed", "output_file": current_latest}
            )

            # Update our known latest
            if content_type == ContentType.IMAGE:
                latest_known_image = current_latest
            else:
                latest_known_video = current_latest

            print(
                f"Successfully associated new {content_type.value} with job {oldest_job_id}"
            )

        return gr.update(value=current_latest if current_latest else None)

    except Exception as e:
        print(f"Error checking for new content: {str(e)}")
        print(f"Stack trace: ", traceback.format_exc())
        return gr.update(value=None)


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
        if len(devices) > 0:
            vram_free = devices[0].get("vram_free")
            vram_total = devices[0].get("vram_total")
            if vram_total > 0:
                vram_used = (vram_total - vram_free) / vram_total

        # print(f"vram_used: {vram_used}")

        if vram_used != previous_vram_used:
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

        # print(f"queue_progress: {queue_finished} out of {queue_total}")

        current_queue_info = [
            queue_finished,
            queue_total,
            queue_failed,
            queue_history,
            len(queue_pending),
            len(queue_running),
        ]
        if queue_total < 1:
            progress(progress=0.0)
            # return gr.update(visible=False)
        elif current_queue_info != previous_queue_info:
            previous_queue_info = current_queue_info
            progress(progress=(queue_finished, queue_total), unit="generations")
            # return gr.update(visible=True)

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

            progress_tuple = (
                current_progress_data.get("value", 0),
                current_progress_data.get("max", 0),
            )
            prompt_id = current_progress_data.get("prompt_id", "N/A")
            if prompt_id != "N/A":
                prompt_id_short = prompt_id[:8]
                progress(progress=progress_tuple, unit="steps", desc=prompt_id_short)
            else:
                progress(progress=0.0)
            # return gr.update(visible=True)
        except:
            progress(progress=0.0)
            # return gr.update(visible=False)

        if not ws or not ws.connected:
            break

        time.sleep(1.0)


def check_gen_progress_visibility():
    """Check if generation progress should be visible"""
    try:
        [queue_running, queue_pending, queue_failed] = get_queue()
        current_step = current_progress_data.get("value", None)
        # Check if there are any running jobs by looking at length of queue_running
        visibility = (current_step is not None) and len(queue_running) > 0
    except:
        visibility = False
    return gr.update(visible=visibility)


def check_interrupt_visibility():
    """Check if queue info should be visible"""
    try:
        [queue_running, queue_pending, queue_failed] = get_queue()
        # Check lengths of running and pending queues
        visibility = len(queue_running) > 0 or len(queue_pending) > 0
    except:
        visibility = False
    return gr.update(visible=visibility)


# endregion


def run_workflow(workflow_filename, workflow_name, progress, **kwargs):
    """Run a workflow with the given parameters"""
    try:
        print(f"\nAttempting to run workflow: {workflow_filename}")
        # print("inside run workflow with kwargs: " + str(kwargs))

        # Construct the path to the workflow JSON file
        workflow_json_filepath = "./workflows/" + workflow_filename

        # Open the workflow JSON file
        try:
            with open(workflow_json_filepath, "r", encoding="utf-8") as file:
                workflow_data = json.load(file)
                print("Successfully loaded workflow JSON")
        except FileNotFoundError:
            print(f"ERROR: Workflow file not found at {workflow_json_filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in workflow file: {str(e)}")
            return None

        try:
            # Iterate through changes requested via kwargs
            for change_request in kwargs.values():
                # Extract the node path and the new value from the change request
                node_path = change_request["node-id"]
                new_value = change_request["value"]

                # Log the intended change for debugging
                print(f"Intending to change {node_path} to {new_value}")

                # Process the node path into a list of keys
                path_keys = node_path.strip("[]").split("][")
                path_keys = [key.strip('"') for key in path_keys]

                # Navigate through the workflow data to the last key
                current_section = workflow_data
                for key in path_keys[:-1]:  # Exclude the last key for now
                    current_section = current_section[key]

                # Update the value at the final key
                final_key = path_keys[-1]
                print(f"Updating {current_section[final_key]} to {new_value}")
                current_section[final_key] = new_value

        except Exception as e:
            print(f"ERROR: Failed to update parameter {change_request}: {str(e)}")
            return None

        print("\nSubmitting workflow to ComfyUI...")
        prompt_id = post_prompt(workflow_data)
        print(f"post_prompt returned: {prompt_id}")

        if not prompt_id:
            print("ERROR: No prompt ID returned from post_prompt")
            return None

        # Track the job
        track_generation_job(prompt_id, workflow_name)
        print(f"Successfully tracked job {prompt_id} for {workflow_name}")
        return prompt_id

    except Exception as e:
        print(f"ERROR in run_workflow: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        return None


def get_job_result(prompt_id):
    """Get the current status and result for a job"""
    if prompt_id in job_tracking:
        job = job_tracking[prompt_id]
        return {
            "status": job["status"],
            "output_file": job["output_file"],
            "workflow_name": job["workflow_name"],
            "timestamp": job["timestamp"],
        }
    return {"status": "not_found"}


def run_workflow_with_name(
    workflow_filename,
    workflow_name,
    raw_components,
    component_info_dict,
    progress=gr.Progress(track_tqdm=True),
):
    output_type = workflow_definitions[workflow_name]["outputs"].get("type", "")

    def wrapper(*args):
        global current_output_type
        current_output_type = output_type  # Set the type when workflow runs
        print(f"I just got told to make a(n) {current_output_type}")

        # match the component to the arg
        for component, arg in zip(raw_components, args):
            # check if component is path type and convert to absolute path if needed
            if component_info_dict[component.elem_id].get("type") == "path" and arg:
                arg = os.path.abspath(arg)

            # access the component_info_dict using component.elem_id and add a value field = arg
            component_info_dict[component.elem_id]["value"] = arg

        return run_workflow(
            workflow_filename,
            workflow_name,
            progress,
            **component_info_dict,
        )

    # Set descriptive name and docstring
    wrapper.__name__ = workflow_name
    wrapper.__doc__ = workflow_definitions[workflow_name].get("description", "")

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
        # print(f"Reveal option input: {selected_option}")
        selected_index = choices.index(selected_option)
        updaters[selected_index] = gr.update(visible=True)

    return updaters


def process_dynamic_input(
    selected_option, possible_options, input_type, *args, progress=gr.Progress()
):
    print("\nProcessing dynamic input")
    print(f"Selected Option: {selected_option}")
    print(f"Possible Options: {possible_options}")

    arg_index = 0

    # Handle limit controls based on input_type
    if input_type == "images":
        limit_enabled = args[arg_index]
        limit_value = args[arg_index + 1]
        arg_index += 2
        print(f"Limit Enabled: {limit_enabled}")
        print(f"Limit Value: {limit_value if limit_enabled else 'No limit'}")
    else:
        limit_enabled = False
        limit_value = None

    option_values = args[arg_index : arg_index + len(possible_options)]
    arg_index += len(possible_options)
    print(f"Option Values: {option_values}")

    selected_index = possible_options.index(selected_option)
    selected_value = option_values[selected_index]

    result = None
    if input_type == "images":
        max_images = int(limit_value) if limit_enabled else None
        if selected_option == "filepath":
            result = organise_local_files(
                selected_value, input_type, max_images=max_images
            )
        elif selected_option == "nilor collection":
            result = resolve_online_collection(
                selected_value, max_images=max_images, progress=progress
            )
        elif selected_option == "upload":
            result = copy_uploaded_files_to_local_dir(
                selected_value, input_type, max_files=max_images
            )
    elif input_type == "video":
        if selected_option == "filepath":
            result = process_video_file(selected_value)
        elif selected_option == "upload":
            result = process_video_file(selected_value)

    # Return values for all components
    return list(option_values) + [result]


def create_dynamic_input(
    input_type, choices, tooltips, text_label, identifier, additional_information=None
):
    markdown_text = f"##### {input_type.capitalize()} Input"
    if additional_information is not None:
        markdown_text += f"\n*{additional_information}*"

    gr.Markdown(markdown_text, elem_classes="group-label")
    with gr.Group():
        selected_option = gr.Radio(choices, label=text_label, value=choices[0])
        print(f"Choices: {choices}")

        # Add optional limit controls only for image inputs
        if input_type == "images":
            with gr.Accordion("Advanced Options", open=False):
                limit_enabled = gr.Checkbox(label="Limit number of images", value=False)
                limit_value = gr.Number(
                    label="Max images",
                    value=4,
                    minimum=1,
                    step=1,
                    interactive=True,
                    visible=True,
                )
        else:
            limit_enabled = None
            limit_value = None

        # Initialize possible_inputs based on input_type
        if input_type == "images":
            possible_inputs = [
                gr.Textbox(
                    label=choices[0], show_label=False, visible=True, info=tooltips[0]
                ),
                gr.Textbox(
                    label=choices[1], show_label=False, visible=False, info=tooltips[1]
                ),
                gr.Gallery(label=choices[2], show_label=False, visible=False),
            ]
        elif input_type == "video":
            possible_inputs = [
                gr.Textbox(
                    label=choices[0], show_label=False, visible=True, info=tooltips[0]
                ),
                gr.File(
                    label=choices[1],
                    show_label=False,
                    visible=False,
                    file_count="single",
                    type="filepath",
                    file_types=["video"],
                ),
            ]

        output = gr.Textbox(
            label="File Path",
            interactive=False,
            elem_id=identifier,
            info="Preview of the file path",
        )

        # Modify visibility of inputs based on selected_option
        selected_option.change(
            select_dynamic_input_option,
            inputs=[selected_option, gr.State(choices)],
            outputs=possible_inputs,
        )

        # Handle input submission with progress tracking
        def process_with_progress(*args, progress=gr.Progress()):
            print(f"Args: {args}")

            selected_opt = args[0]  # First arg is selected option
            choices_state = args[1]  # Second arg is choices State
            input_type_state = args[2]  # Third arg is input_type State

            # Initialize variables
            max_images = None
            input_values = []

            arg_index = 3  # Starting index for additional arguments

            # If limit controls exist, they come next
            if limit_enabled and limit_value:
                limit_enabled_value = args[arg_index]
                limit_value_value = args[arg_index + 1]
                arg_index += 2  # Increment index since we have consumed two arguments

                print(f"Extracted limit_enabled_value: {limit_enabled_value}")
                print(f"Extracted limit_value_value: {limit_value_value}")

                max_images = int(limit_value_value) if limit_enabled_value else None
            else:
                limit_enabled_value = None
                limit_value_value = None

            # The rest of the args are input values
            input_values = list(args[arg_index:])

            # Debugging output
            print(f"Selected Option: {selected_opt}")
            print(f"Limit Enabled: {limit_enabled_value}")
            print(f"Limit Value: {limit_value_value}")
            print(f"Input Values: {input_values}")

            # Get the selected input value based on the selected option
            opt_index = choices.index(selected_opt)
            if opt_index < len(input_values):
                selected_value = input_values[opt_index]
            else:
                print(
                    f"Error: Index {opt_index} out of range for input_values with length {len(input_values)}"
                )
                selected_value = None  # Handle this case appropriately

            if selected_opt == "nilor collection":
                progress(0, desc="Requesting collection...")
                try:
                    result = resolve_online_collection(
                        selected_value, max_images=max_images, progress=progress
                    )
                except Exception as e:
                    result = None
                    print(f"Failed to resolve online collection: {e}")
                    gr.Warning(f"Error resolving collection: {e}")
            elif selected_opt == "filepath":
                result = organise_local_files(
                    selected_value,
                    input_type_state,
                    max_images=max_images,
                    shuffle=False,
                )
            elif selected_opt == "upload":
                result = copy_uploaded_files_to_local_dir(
                    selected_value,
                    input_type_state,
                    max_files=max_images,
                    shuffle=False,
                )
            else:
                result = process_dynamic_input(
                    selected_opt, choices_state, input_type_state, *input_values
                )

            # Return all input values plus result
            return input_values + [result]

        # Prepare the inputs list for the submit/upload events
        # Only include limit controls if they are not None
        optional_inputs = []
        if limit_enabled and limit_value:
            optional_inputs.extend([limit_enabled, limit_value])

        event_inputs = (
            [selected_option, gr.State(choices), gr.State(input_type)]
            + optional_inputs
            + possible_inputs
        )

        # Handle input submission
        for input_box in possible_inputs:
            if isinstance(input_box, gr.Textbox):
                input_box.submit(
                    fn=process_with_progress,
                    inputs=event_inputs,
                    outputs=possible_inputs + [output],
                )
            elif isinstance(input_box, (gr.File, gr.Gallery)):
                input_box.upload(
                    fn=process_with_progress,
                    inputs=event_inputs,
                    outputs=possible_inputs + [output],
                )

        return selected_option, possible_inputs, output


# Ensure all elements in self.inputs are valid Gradio components
def filter_valid_components(components):
    valid_components = []
    for component in components:
        if hasattr(component, "_id"):
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
    resetter_visibility = False
    if component == default_value:
        html = ""
        resetter_visibility = False
    else:
        html = f"<style>#{elem_id} {{ background: var(--background-fill-secondary); }}</style>"
        resetter_visibility = True

    # Return updates with display="none"
    return gr.update(
        value=html, visible=False, elem_classes="hide-container"
    ), gr.update(visible=resetter_visibility)


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
    input_choices = input_details.get("choices", None)
    input_info = input_details.get("info", None)

    # Define a mapping of input types to Gradio components
    component_map = {
        "path": gr.Textbox,
        "string": gr.Textbox,
        "text": gr.Textbox,
        "images": None,  # special case for radio selection handled below
        "video": None,  # special case for video selection handled below
        "bool": gr.Checkbox,
        "float": gr.Number,
        "int": gr.Number,
        "slider": gr.Slider,
        "radio": gr.Radio,  # True radios collect their options from the workflow_definitions.json
        "enum": gr.Dropdown,
        "group": None,
        "toggle-group": gr.Checkbox,
    }

    component = None
    reset_button = None
    components = []
    components_dict = {}

    with gr.Group():
        if input_type in component_map:
            # Use the mapping to find Gradio component based on input_type
            component_constructor = component_map.get(input_type)

            if input_type == "group":
                gr.Markdown(f"##### {input_label}", elem_classes="group-label")

                with gr.Group():
                    # Group of inputs
                    with gr.Group():
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(
                                sub_context, group_input_key
                            )

                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)
            elif input_type == "toggle-group":
                with gr.Group():
                    with gr.Row(equal_height=True):
                        # Checkbox component which enables Group
                        component = component_constructor(
                            label=input_label,
                            elem_id=input_key,
                            value=input_value,
                            interactive=input_interactive,
                            scale=100,
                            info=input_info,
                        )

                        # Compact Reset button with reduced width, initially hidden
                        reset_button = gr.Button(
                            "↺",
                            visible=False,
                            elem_id="reset-button",
                            scale=1,
                            variant="secondary",
                            min_width=5,
                        )

                    # Group of inputs (initially hidden)
                    with gr.Group(visible=component.value) as input_group:
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(
                                sub_context, group_input_key
                            )

                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)

                # Update the group visibility based on the checkbox
                component.change(
                    fn=toggle_group,
                    inputs=component,
                    outputs=input_group,
                    queue=False,
                    show_progress="hidden",
                )
            elif input_type == "images":
                # print("!!!!!!!!!!!!!!!!!!!!!!!\nMaking Radio")
                selected_option, inputs, component = create_dynamic_input(
                    input_type,
                    choices=["filepath", "nilor collection", "upload"],
                    tooltips=[
                        "Enter the path of the directory of images and press Enter to submit",
                        "Enter the name of the Nilor Collection and press Enter to resolve",
                    ],
                    text_label="Select Input Type",
                    identifier=input_key,
                    additional_information=input_info,
                )
            elif input_type == "video":
                selected_option, inputs, component = create_dynamic_input(
                    input_type,
                    choices=["filepath", "upload"],
                    tooltips=[
                        "Enter the path of the directory of video and press Enter to submit"
                    ],
                    text_label="Select Input Type",
                    identifier=input_key,
                    additional_information=input_info,
                )
            elif input_type == "float" or input_type == "int" or input_type == "slider":
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
                        value=input_value,
                        minimum=input_minimum,
                        maximum=input_maximum,
                        step=input_step,
                        interactive=input_interactive,
                        scale=100,
                        info=input_info,
                    )

                    if input_type != "slider":  # not required for slider
                        # Compact Reset button with reduced width, initially hidden
                        reset_button = gr.Button(
                            "↺",
                            visible=False,
                            elem_id="reset-button",
                            scale=1,
                            variant="secondary",
                            min_width=5,
                        )
            elif input_type == "radio":
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
                        choices=input_choices,
                        value=input_value,
                        scale=100,
                        info=input_info,
                    )

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button(
                        "↺",
                        visible=False,
                        elem_id="reset-button",
                        scale=1,
                        variant="secondary",
                        min_width=5,
                    )
            elif input_type == "enum":
                with gr.Row(equal_height=True):
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
                        choices=input_choices,
                        value=input_value,
                        interactive=input_interactive,
                        scale=100,
                        info=input_info,
                    )
                    reset_button = gr.Button(
                        "↺",
                        visible=False,
                        elem_id="reset-button",
                        scale=1,
                        variant="secondary",
                        min_width=5,
                    )
            else:
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
                        value=input_value,
                        interactive=input_interactive,
                        scale=100,
                        info=input_info,
                    )

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button(
                        "↺",
                        visible=False,
                        elem_id="reset-button",
                        scale=1,
                        variant="secondary",
                        min_width=5,
                    )

            if component is not None:
                components.append(component)
                components_dict[input_key] = input_details

                if reset_button is not None:
                    # Trigger the reset check when the value of the input changes
                    html_output = gr.HTML(visible=False)
                    component.change(
                        fn=watch_input,
                        inputs=[component, gr.State(input_value), gr.State(input_key)],
                        outputs=[html_output, reset_button],
                        queue=False,
                        show_progress="hidden",
                    )

                    # Trigger the reset function when the button is clicked
                    reset_button.click(
                        fn=reset_input,
                        inputs=[gr.State(input_value)],
                        outputs=component,
                        queue=False,
                        show_progress="hidden",
                    )
        else:
            print(f"Whoa! Unsupported input type: {input_type}")

    return [components, components_dict]
    # return components


def apply_preset(workflow_name, preset_name):
    if preset_name == "Default":
        # Return default values from workflow definition
        defaults = {
            key: details["value"]
            for key, details in workflow_definitions[workflow_name]["inputs"].items()
        }
        return [defaults[comp.elem_id] for comp in components]

    # Get preset values
    preset = workflow_definitions[workflow_name]["presets"][preset_name]
    preset_values = preset["values"]

    # Update component values
    updates = []
    for component in components:
        if component.elem_id in preset_values:
            updates.append(preset_values[component.elem_id])
        else:
            # Keep existing value if not in preset
            updates.append(component.value)

    return updates


def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")

    components = []  # Initialize components list
    component_data_dict = {}

    # Add preset selector if workflow has presets
    presets = workflow_definitions[workflow_name].get("presets", {})
    preset_dropdown = None
    if presets:
        preset_choices = ["Default"] + list(presets.keys())
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=preset_choices,
                value="Default",
                label="Presets",
                info="Select a preset to automatically configure multiple parameters",
            )

    key_context = workflow_definitions[workflow_name]["inputs"]

    interactive_inputs = []
    noninteractive_inputs = []

    # Sort inputs by interactivity
    for input_key in key_context:
        input_details = key_context[input_key]
        input_interactive = input_details.get("interactive", True)
        if input_interactive:
            interactive_inputs.append(input_key)
        else:
            noninteractive_inputs.append(input_key)

    # Process interactive inputs
    for input_key in interactive_inputs:
        [sub_components, sub_dict_values] = process_input(key_context, input_key)
        components.extend(sub_components)
        component_data_dict.update(sub_dict_values)

    # Process non-interactive inputs
    if noninteractive_inputs:
        with gr.Accordion("Constants", open=False):
            gr.Markdown(
                "You can edit these constants in workflow_definitions.json if you know what you're doing."
            )
            for input_key in noninteractive_inputs:
                [sub_components, sub_dict_values] = process_input(
                    key_context, input_key
                )
                components.extend(sub_components)
                component_data_dict.update(sub_dict_values)

    # Add preset change handler
    if preset_dropdown is not None:

        def apply_preset(preset_name):
            if preset_name == "Default":
                # Return default values from workflow definition
                updates = []
                for comp in components:
                    if hasattr(comp, "elem_id") and comp.elem_id in key_context:
                        updates.append(key_context[comp.elem_id].get("value"))
                    else:
                        updates.append(comp.value)
                return updates

            # Get preset values
            preset_values = presets[preset_name]["values"]

            # Update component values
            updates = []
            for comp in components:
                if hasattr(comp, "elem_id") and comp.elem_id in preset_values:
                    updates.append(preset_values[comp.elem_id])
                else:
                    updates.append(comp.value)

            return updates

        preset_dropdown.change(
            fn=apply_preset, inputs=[preset_dropdown], outputs=components
        )

    return components, component_data_dict


def load_demo():
    global ws, tick_timer, threads
    print("Loading the demo!!!")

    # Close any existing connection
    if ws:
        try:
            ws.close()
        except websocket.WebSocketException as e:
            print(f"Error closing websocket: {e}")
        except ConnectionError as e:
            print(f"Connection error while closing: {e}")
        ws = None

    tick_timer = gr.Timer(value=1.0)
    ws = connect_to_websocket(client_id)

    if ws:
        try:
            # Start threading after connecting
            print(f"WebSocket connection attempt")
            heartbeat_thread = threading.Thread(
                target=send_heartbeat, args=(ws,), daemon=True
            )
            progress_thread = threading.Thread(
                target=check_current_progress, daemon=True
            )

            threads = [heartbeat_thread, progress_thread]
            for thread in threads:
                thread.start()
                print(f"Started thread: {thread.name}")  # Debug log

        except websocket.WebSocketConnectionClosedException as e:
            print(f"WebSocket connection closed: {e}")
            ws = reconnect(ws, client_id)
        except Exception as e:
            print(f"An error occurred: {type(e).__name__}: {e}")

    # Reset state variables...


def unload_demo():
    global ws, tick_timer, check_vram_running, check_queue_running, check_progress_running
    print("Unloading the demo...")

    # Stop all monitoring loops
    check_vram_running = False
    check_queue_running = False
    check_progress_running = False

    # Deactivate timer
    if tick_timer:
        tick_timer.active = False

    # Close WebSocket connection
    if ws:
        try:
            ws.close()
        except websocket.WebSocketException as e:
            print(f"Error closing websocket: {e}")
        except ConnectionError as e:
            print(f"Connection error while closing: {e}")
        ws = None

    time.sleep(2.0)


def setup_signal_handlers():
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
            print("Signal handlers set up successfully")
        except ValueError as e:
            print(f"Could not set up signal handlers: {e}")


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

.logs textarea {
    font-family: monospace;
    font-size: 12px;
    background-color: var(--neutral-950);
    color: var(--neutral-100);
}

.progress-bar {
    background: linear-gradient(to right, var(--secondary-500), var(--primary-500));
}

.progress-text{
    font-family: monospace;
}
"""


def update_download_progress():
    if download_progress["total"] > 0:
        progress = download_progress["current"] / download_progress["total"]
        return gr.update(value=download_progress["status"]), progress


with gr.Blocks(
    title="Zenerator",
    theme=gr.themes.Ocean(font=gr.themes.GoogleFont("DM Sans")),
    css=custom_css,
) as demo:
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
                            "Zenerator is a tool by Nilor Corp for creating and running generative AI workflows.\n\n"
                            "Select a workflow from the tabs above and fill in the parameters.\n\n"
                            "Click 'Run Workflow' to start the workflow.  ",
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
                                        comfy_url_and_port_selector = gr.Dropdown(
                                            label="ComfyUI Port",
                                            choices=COMFY_PORTS,
                                            value=COMFY_PORTS[0],
                                            interactive=True,
                                            scale=1,
                                        )
                                        print(
                                            f"Default ComfyUI Port: {comfy_url_and_port_selector.value}"
                                        )
                                        comfy_url_and_port_selector.change(
                                            select_correct_port,
                                            inputs=[comfy_url_and_port_selector],
                                        )
                                        run_button = gr.Button(
                                            "Run Workflow",
                                            variant="primary",
                                            scale=3,
                                            elem_id="run-button",
                                        )
                                    with gr.Accordion(
                                        "Workflow Info",
                                        open=False,
                                        elem_id="workflow-info",
                                    ):
                                        info = gr.Markdown(
                                            workflow_definitions[workflow_name].get(
                                                "description", ""
                                            )
                                        )

                                # also make a dictionary with the components' data
                                components, component_dict = create_tab_interface(
                                    workflow_name
                                )

                            output_type = workflow_definitions[workflow_name][
                                "outputs"
                            ].get("type", "")

                    run_button.click(
                        fn=run_workflow_with_name(
                            workflow_filename, workflow_name, components, component_dict
                        ),
                        inputs=components,
                        outputs=gr.Text(visible=False),
                        # outputs=[gen_component],
                        trigger_mode="multiple",
                        # show_progress="full"
                        api_name=f"workflow/{workflow_name}",
                    )

        with gr.Column(scale=4):
            # TODO: is it possible to preview only an output that was produced by this workflow tab? otherwise this should probably exist outside of the workflow tab
            gr.Markdown("### Output Preview")
            with gr.Group():
                if output_type == "image":
                    latest_content = get_latest_image(OUT_DIR)
                    if latest_content is not None:
                        output_player = gr.Image(
                            show_label=False, interactive=False, value=latest_content
                        )
                else:
                    # populate the Output Preview with the latest video in the output directory
                    latest_content = get_latest_video(OUT_DIR)
                    if latest_content is not None:
                        output_player = gr.Video(
                            value=latest_content,
                            show_label=False,
                            autoplay=True,
                            loop=True,
                            interactive=False,
                        )
                    else:
                        output_player = gr.Video(
                            show_label=False,
                            autoplay=True,
                            loop=True,
                            interactive=False,
                        )
                # output_filepath_component = gr.Markdown("N/A")

                tick_timer.tick(
                    fn=check_for_new_content,
                    outputs=[output_player],
                    show_progress="hidden",
                )

            with gr.Group(visible=False) as gen_progress_group:
                gen_component = gr.Textbox(
                    label="Generation Progress", interactive=False, visible=True
                )

                tick_timer.tick(
                    fn=check_progress, outputs=gen_component, show_progress="full"
                )

            tick_timer.tick(
                fn=check_gen_progress_visibility,
                outputs=gen_progress_group,
                show_progress="hidden",
            )

            with gr.Group(visible=False) as queue_progress_group:
                queue_info_component = gr.Textbox(
                    label="Queue Info", interactive=False, visible=True
                )

                tick_timer.tick(
                    fn=check_queue, outputs=queue_info_component, show_progress="full"
                )

            tick_timer.tick(
                fn=check_interrupt_visibility,
                outputs=queue_progress_group,
                show_progress="hidden",
            )

            with gr.Group():
                vram_usage_component = gr.Textbox(
                    label="VRAM Usage", interactive=False, visible=True
                )

                tick_timer.tick(
                    fn=check_vram, outputs=[vram_usage_component], show_progress="full"
                )

            with gr.Group() as interrupt_group:
                interrupt_button = gr.Button("Interrupt", visible=False, variant="stop")
                interrupt_button.click(fn=post_interrupt)

            tick_timer.tick(
                fn=check_interrupt_visibility,
                outputs=interrupt_group,
                show_progress="hidden",
            )

    if __name__ == "__main__":
        setup_signal_handlers()
        initialize_content_tracking()
        demo.queue()

        # Create hidden components for the result endpoint
        result_input = gr.Text(visible=False)
        result_output = gr.JSON(visible=False)
        demo.load(
            fn=get_job_result,
            inputs=result_input,
            outputs=result_output,
            api_name="workflow_result",
        )

        demo.launch(allowed_paths=allowed_paths, favicon_path="favicon.png")
