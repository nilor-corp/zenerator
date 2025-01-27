import json
import requests
import os
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
import threading
import websocket
import uuid
import torch
import signal
import sys
from enum import Enum
import traceback
import weakref
from typing import Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor
import gc
import asyncio


# Resource management for better memory handling
class ResourceManager:
    def __init__(self):
        self._websockets: Set[websocket.WebSocket] = weakref.WeakSet()
        self._threads: Set[threading.Thread] = set()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = True

    def add_websocket(self, ws: websocket.WebSocket):
        self._websockets.add(ws)

    def add_thread(self, thread: threading.Thread):
        self._threads.add(thread)

    def cleanup(self):
        self._running = False

        # Clean up websockets
        for ws in list(self._websockets):
            try:
                ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")

        # Clean up threads
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=2.0)

        # Clean up executor
        if self._executor:
            self._executor.shutdown(wait=False)

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AppState:
    def __init__(self):
        self.output_type: str = ""
        self.current_output_type: str = "image"
        self.download_progress: Dict = {"current": 0, "total": 0, "status": ""}
        self.job_tracking: Dict = {}
        self.current_progress_data: Dict = {}
        self.previous_progress_data: Dict = {}
        self.current_queue_info: list = [-1] * 6
        self.previous_queue_info: list = [-1] * 6
        self.check_current_progress_running: bool = False
        self.latest_known_image: Optional[str] = None
        self.latest_known_video: Optional[str] = None
        self.check_vram_running: bool = False
        self.check_queue_running: bool = False
        self.check_progress_running: bool = False
        self.previous_vram_used: float = -1.0
        self.websocket_manager = None


# Global instances
resource_manager = ResourceManager()
app_state = AppState()

# Load config
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

for port in COMFY_PORTS:
    QUEUE_URLS.append(f"http://{COMFY_IP}:{port}")

selected_port_url = QUEUE_URLS[0]

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

check_current_progress_running = False
current_progress_data = {}
previous_progress_data = {}
current_queue_info = [-1, -1, -1, -1, -1, -1]
previous_queue_info = [-1, -1, -1, -1, -1, -1]


class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"


def signal_handler(signum, frame):
    global running, tick_timer, threads
    print("\nShutdown signal received. Cleaning up...")
    running = False

    # Stop all monitoring loops
    app_state.check_vram_running = False
    app_state.check_queue_running = False
    app_state.check_progress_running = False

    # Deactivate timer
    if tick_timer:
        tick_timer.active = False
    tick_timer = None

    # Clean up resources
    resource_manager.cleanup()

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
class WebSocketManager:
    def __init__(self):
        self.ws = None
        self.client_id = str(uuid.uuid4())

    def connect(self):
        self.ws = websocket.WebSocket()
        try:
            self.ws.connect(
                f"ws://{config['COMFY_IP']}:{config['COMFY_PORTS'][0]}/ws?clientId={self.client_id}"
            )
            resource_manager.add_websocket(self.ws)
        except ConnectionResetError as e:
            print(f"Connection was reset: {e}")
            return None
        except Exception as e:
            print(f"Exception while connecting: {e}")
            return None

        print("Connected to WebSocket successfully!")
        return self.ws

    def reconnect(self, max_retries=5):
        retries = 0
        while retries < max_retries:
            print(f"Attempting to reconnect ({retries + 1}/{max_retries})...")
            if self.connect():
                return self.ws
            retries += 1
            time.sleep(1)
        print("Max retries reached. Could not reconnect.")
        return None


def send_heartbeat(ws):
    while ws and ws.connected and resource_manager._running:
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


def check_current_progress(ws):
    """Monitor workflow progress with proper state management"""
    app_state.check_current_progress_running = True
    print("Starting progress check thread")

    while app_state.check_current_progress_running and ws and ws.connected:
        try:
            out = ws.recv()
            if not out:
                continue

            if isinstance(out, str):
                try:
                    message = json.loads(out)

                    if message["type"] in ["progress", "executing"]:
                        app_state.current_progress_data = message["data"]
                    elif message["type"] == "status":
                        if "status" in message["data"]:
                            status_data = message["data"]["status"]
                            if "exec_info" in status_data:
                                app_state.current_progress_data = status_data[
                                    "exec_info"
                                ]
                except json.JSONDecodeError:
                    print("Received invalid JSON data, skipping...")
                    continue
            else:
                continue

        except websocket.WebSocketConnectionClosedException:
            print("WebSocket connection closed, stopping progress check")
            break
        except Exception as e:
            print(f"Error in progress check: {type(e).__name__}: {e}")
            break

    app_state.check_current_progress_running = False
    app_state.current_progress_data = {}


# Create websocket manager instance
websocket_manager = WebSocketManager()
app_state.websocket_manager = websocket_manager

# endregion


# region POST Requests
def comfy_POST(endpoint, message):
    post_url = selected_port_url + "/" + endpoint
    data = json.dumps(message).encode("utf-8")
    print(f"POST {endpoint} on: {post_url}")
    try:
        post_response = requests.post(post_url, data=data)
        return post_response
    except ConnectionResetError:
        print("Connection was reset while trying to start the workflow. Retrying...")
    except requests.RequestException as e:
        print(f"Error querying the GET endpoint {endpoint}: ", e)


def post_prompt(workflow):
    """Submit a workflow prompt to ComfyUI"""
    prompt_data = {
        "prompt": workflow,
        "client_id": app_state.websocket_manager.client_id,
    }

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
    """Interrupt current processing with proper state management"""
    app_state.current_progress_data = {}
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
    try:
        prompt = comfy_GET("prompt")
        if prompt is None:
            print("/prompt GET response is empty")
            return "N/A"

        status = prompt.get("status", "N/A")
        print(f"Prompt endpoint data: {prompt}")
        return status
    except Exception as e:
        print(f"Error getting prompt status: {e}")
        return "N/A"


def get_history():
    """Get processing history"""
    history = comfy_GET("history")
    if history is None:
        print("/history GET response is empty")
        return {}
    return history


def get_system_stats():
    """Get system statistics with proper error handling"""
    try:
        system_stats = comfy_GET("system_stats")
        if system_stats is None:
            print("/system_stats GET response is empty")
            return [[], []]

        devices = system_stats.get("devices")
        if devices is None:
            return [system_stats, []]

        return [system_stats, devices]
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return [[], []]


# endregion

with open("workflow_definitions.json") as f:
    workflow_definitions = json.load(f)


def get_lora_filenames(directory):
    files = os.listdir(directory)
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
    image_files = get_all_images(folder)
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


def get_latest_image_with_prefix(folder, prefix):
    image_files = get_all_images(folder)
    image_files = [f for f in image_files if f.lower().startswith(prefix)]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


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
    app_state.job_tracking[job_id] = {
        "status": "pending",
        "type": content_type,
        "workflow_name": workflow_name,
        "timestamp": time.time(),
        "output_file": None,
    }


def initialize_content_tracking():
    """Record the latest content files at startup"""
    print("\nInitializing content tracking...")
    app_state.latest_known_image = get_latest_content(OUT_DIR, ContentType.IMAGE.value)
    app_state.latest_known_video = get_latest_content(OUT_DIR, ContentType.VIDEO.value)
    print(f"Initial image: {app_state.latest_known_image}")
    print(f"Initial video: {app_state.latest_known_video}")


def check_for_new_content():
    """Check for new content and associate with correct job"""
    try:
        # Only log job tracking when there are NEW pending jobs
        pending_jobs_now = {
            pid
            for pid, job in app_state.job_tracking.items()
            if job["status"] == "pending"
        }

        if not hasattr(check_for_new_content, "last_pending_jobs"):
            check_for_new_content.last_pending_jobs = set()

        if pending_jobs_now != check_for_new_content.last_pending_jobs:
            print(f"\nPending jobs changed: {len(pending_jobs_now)} jobs waiting")
            print(f"Current job tracking: {app_state.job_tracking}")
            check_for_new_content.last_pending_jobs = pending_jobs_now

        for content_type in ContentType:
            current_latest = get_latest_content(OUT_DIR, content_type.value)
            if not current_latest:
                continue

            known_latest = (
                app_state.latest_known_image
                if content_type == ContentType.IMAGE
                else app_state.latest_known_video
            )

            if current_latest == known_latest:
                continue

            print(f"\nFound new {content_type.value}: {current_latest}")

            pending_jobs = {
                pid: data
                for pid, data in app_state.job_tracking.items()
                if data["status"] == "pending"
                and data["type"] == content_type
                and data["output_file"] is None
            }

            if not pending_jobs:
                if content_type == ContentType.IMAGE:
                    app_state.latest_known_image = current_latest
                else:
                    app_state.latest_known_video = current_latest
                print(
                    f"No pending jobs for new {content_type.value}, updated known latest"
                )
                continue

            oldest_job_id = min(
                pending_jobs.keys(), key=lambda pid: pending_jobs[pid]["timestamp"]
            )

            app_state.job_tracking[oldest_job_id].update(
                {"status": "completed", "output_file": current_latest}
            )

            if content_type == ContentType.IMAGE:
                app_state.latest_known_image = current_latest
            else:
                app_state.latest_known_video = current_latest

            print(
                f"Successfully associated new {content_type.value} with job {oldest_job_id}"
            )

        return gr.update(value=current_latest if current_latest else None)

    except Exception as e:
        print(f"Error checking for new content: {str(e)}")
        print(f"Stack trace: ", traceback.format_exc())
        return gr.update(value=None)


def check_vram(progress=gr.Progress()):
    app_state.check_vram_running = True

    while app_state.check_vram_running:
        try:
            [system_stats, devices] = get_system_stats()

            vram_used = 0.0
            if len(devices) > 0:
                vram_free = devices[0].get("vram_free")
                vram_total = devices[0].get("vram_total")
                if vram_total > 0:
                    vram_used = (vram_total - vram_free) / vram_total

            if vram_used != app_state.previous_vram_used:
                progress(progress=vram_used)
                app_state.previous_vram_used = vram_used

            time.sleep(1.0)
        except Exception as e:
            print(f"Error checking VRAM: {e}")
            time.sleep(1.0)


def check_queue(progress=gr.Progress()):
    app_state.check_queue_running = True

    while app_state.check_queue_running:
        try:
            [queue_running, queue_pending, queue_failed] = get_queue()
            queue_history = get_history()

            queue_finished = len(queue_history)
            queue_total = queue_finished + len(queue_running) + len(queue_pending)

            current_info = [
                queue_finished,
                queue_total,
                queue_failed,
                queue_history,
                len(queue_pending),
                len(queue_running),
            ]

            if queue_total < 1:
                progress(progress=0.0)
            elif current_info != app_state.current_queue_info:
                app_state.current_queue_info = current_info
                progress(progress=(queue_finished, queue_total), unit="generations")

            time.sleep(1.0)
        except Exception as e:
            print(f"Error checking queue: {e}")
            time.sleep(1.0)


def check_progress(progress=gr.Progress()):
    try:
        if app_state.current_progress_data:
            value = app_state.current_progress_data.get("value", 0)
            max_value = app_state.current_progress_data.get("max", 1)
            if max_value > 0:
                progress(progress=(value, max_value), unit="steps")
        else:
            progress(progress=0.0)
    except Exception as e:
        print(f"Error in check_progress: {e}")
        progress(progress=0.0)


def check_gen_progress_visibility():
    try:
        prompt_id = app_state.current_progress_data.get("prompt_id", None)
        current_step = app_state.current_progress_data.get("value", None)
        queue_running = app_state.current_queue_info[5]
        visibility = (current_step is not None) and (queue_running > 0)
    except Exception:
        visibility = False
    return gr.update(visible=visibility)


def check_interrupt_visibility():
    """Check if queue info should be visible"""
    try:
        [queue_running, queue_pending, queue_failed] = get_queue()
        visibility = len(queue_running) > 0 or len(queue_pending) > 0
    except Exception:
        visibility = False
    return gr.update(visible=visibility)


# endregion


def run_workflow(workflow_filename, workflow_name, progress, **kwargs):
    """Run a workflow with the given parameters"""
    try:
        print(f"\nAttempting to run workflow: {workflow_filename}")

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


async def delayed_cleanup(prompt_id: str):
    await asyncio.sleep(30)  # Non-blocking 30 second buffer
    if prompt_id in app_state.job_tracking:
        del app_state.job_tracking[prompt_id]
        print(f"Cleaned up completed job {prompt_id} from tracking")


def get_job_result(prompt_id):
    """Get the current status and result for a job"""
    if prompt_id in app_state.job_tracking:
        job = app_state.job_tracking[prompt_id]

        response = {
            "status": job["status"],
            "output_file": job["output_file"],
            "workflow_name": job["workflow_name"],
            "timestamp": job["timestamp"],
        }

        # Schedule cleanup if job is done and we have the output file (or it failed)
        if (job["status"] == "completed" and job["output_file"]) or job[
            "status"
        ] == "failed":
            asyncio.create_task(delayed_cleanup(prompt_id))

        return response

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
        app_state.current_output_type = output_type
        print(f"I just got told to make a(n) {app_state.current_output_type}")

        for component, arg in zip(raw_components, args):
            if component_info_dict[component.elem_id].get("type") == "path" and arg:
                arg = os.path.abspath(arg)
            component_info_dict[component.elem_id]["value"] = arg

        return run_workflow(
            workflow_filename,
            workflow_name,
            progress,
            **component_info_dict,
        )

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
    updaters = [gr.update(visible=False) for _ in choices]

    if selected_option in choices:
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
        elif input_type == "image":  # Single image type
            possible_inputs = [
                gr.Textbox(
                    label=choices[0], show_label=False, visible=True, info=tooltips[0]
                ),
                gr.Image(
                    label=choices[1],
                    show_label=False,
                    visible=False,
                    type="filepath",
                    sources=["upload"],
                ),
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

        # Handle input submission
        def process_with_progress(*args, progress=gr.Progress()):
            print(f"Processing inputs with args: {args}")

            selected_opt = args[0]
            choices_state = args[1]
            input_type_state = args[2]
            arg_index = 3

            if limit_enabled and limit_value:
                limit_enabled_value = args[arg_index]
                limit_value_value = args[arg_index + 1]
                arg_index += 2
                print(
                    f"Limit settings - Enabled: {limit_enabled_value}, Value: {limit_value_value}"
                )
            else:
                limit_enabled_value = None
                limit_value_value = None

            input_values = list(args[arg_index:])
            print(f"Input values: {input_values}")

            try:
                if selected_opt == "nilor collection":
                    progress(0, desc="Requesting collection...")
                    result = resolve_online_collection(
                        input_values[1],  # Use the nilor collection input
                        max_images=(
                            int(limit_value_value) if limit_enabled_value else None
                        ),
                        progress=progress,
                    )
                elif selected_opt == "filepath":
                    result = organise_local_files(
                        input_values[0],  # Use the filepath input
                        input_type_state,
                        max_images=(
                            int(limit_value_value) if limit_enabled_value else None
                        ),
                    )
                elif selected_opt == "upload":
                    result = copy_uploaded_files_to_local_dir(
                        input_values[-1],  # Use the upload input
                        input_type_state,
                        max_files=(
                            int(limit_value_value) if limit_enabled_value else None
                        ),
                    )
                else:
                    result = None
                    print(f"Unknown option selected: {selected_opt}")

                return input_values + [result]

            except Exception as e:
                print(f"Error processing input: {e}")
                return input_values + [None]

        # Prepare inputs for the events
        event_inputs = [selected_option, gr.State(choices), gr.State(input_type)]
        if limit_enabled and limit_value:
            event_inputs.extend([limit_enabled, limit_value])
        event_inputs.extend(possible_inputs)

        # Add events for each input type
        for input_box in possible_inputs:
            if isinstance(input_box, gr.Textbox):
                input_box.submit(
                    fn=process_with_progress,
                    inputs=event_inputs,
                    outputs=possible_inputs + [output],
                )
            elif isinstance(input_box, (gr.File, gr.Gallery, gr.Image)):
                input_box.upload(
                    fn=process_with_progress,
                    inputs=event_inputs,
                    outputs=possible_inputs + [output],
                )

        return selected_option, possible_inputs, output


# Ensure all elements are valid Gradio components
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
        "image": None,  # special case for image selection handled below
        "images": None,  # special case for images selection handled below
        "video": None,  # special case for video selection handled below
        "bool": gr.Checkbox,
        "float": gr.Number,
        "int": gr.Number,
        "slider": gr.Slider,
        "radio": gr.Radio,
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
            component_constructor = component_map.get(input_type)

            if input_type == "group":
                gr.Markdown(f"##### {input_label}", elem_classes="group-label")
                with gr.Group():
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
                        component = component_constructor(
                            label=input_label,
                            elem_id=input_key,
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

                    with gr.Group(visible=component.value) as input_group:
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(
                                sub_context, group_input_key
                            )
                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)

                component.change(
                    fn=toggle_group,
                    inputs=component,
                    outputs=input_group,
                    queue=False,
                    show_progress="hidden",
                )

            elif input_type == "images":
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
            elif input_type == "image":
                selected_option, inputs, component = create_dynamic_input(
                    input_type,
                    choices=["filepath", "upload"],
                    tooltips=[
                        "Enter the path to an image file and press Enter to submit",
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
            elif input_type in ["float", "int", "slider"]:
                with gr.Row(equal_height=True):
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

                    if input_type != "slider":
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
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
                        choices=input_choices,
                        value=input_value,
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
                    component = component_constructor(
                        label=input_label,
                        elem_id=input_key,
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

            if component is not None:
                components.append(component)
                components_dict[input_key] = input_details

                if reset_button is not None:
                    html_output = gr.HTML(visible=False)
                    component.change(
                        fn=watch_input,
                        inputs=[component, gr.State(input_value), gr.State(input_key)],
                        outputs=[html_output, reset_button],
                        queue=False,
                        show_progress="hidden",
                    )

                    reset_button.click(
                        fn=reset_input,
                        inputs=[gr.State(input_value)],
                        outputs=component,
                        queue=False,
                        show_progress="hidden",
                    )

    return [components, components_dict]


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
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=components,
        )

    return components, component_data_dict


def load_demo():
    """Initialize demo with proper resource management"""
    global tick_timer, threads
    print("Loading the demo!!!")

    # Clean up any existing connection
    if app_state.websocket_manager.ws:
        try:
            app_state.websocket_manager.ws.close()
        except websocket.WebSocketException as e:
            print(f"Error closing websocket: {e}")
        except ConnectionError as e:
            print(f"Connection error while closing: {e}")
        app_state.websocket_manager.ws = None

    tick_timer = gr.Timer(value=1.0)
    app_state.websocket_manager.connect()

    if app_state.websocket_manager.ws:
        try:
            # Start threading after connecting
            print(f"WebSocket connection attempt")
            heartbeat_thread = threading.Thread(
                target=send_heartbeat,
                args=(app_state.websocket_manager.ws,),
                daemon=True,
            )
            progress_thread = threading.Thread(
                target=check_current_progress,
                args=(app_state.websocket_manager.ws,),
                daemon=True,
            )

            # Add threads to resource manager for cleanup
            resource_manager.add_thread(heartbeat_thread)
            resource_manager.add_thread(progress_thread)

            threads = [heartbeat_thread, progress_thread]
            for thread in threads:
                thread.start()
                print(f"Started thread: {thread.name}")

        except Exception as e:
            print(f"An error occurred: {type(e).__name__}: {e}")


def unload_demo():
    global ws, tick_timer
    print("Unloading the demo...")

    # Stop all monitoring loops
    app_state.check_vram_running = False
    app_state.check_queue_running = False
    app_state.check_progress_running = False

    # Deactivate timer
    if tick_timer:
        tick_timer.active = False

    # Clean up resources
    resource_manager.cleanup()

    time.sleep(2.0)


def setup_signal_handlers():
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
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
    overflow-y: scroll;
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

                    with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):
                        with gr.Row():
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

                                components, component_dict = create_tab_interface(
                                    workflow_name
                                )

                            output_type = workflow_definitions[workflow_name][
                                "outputs"
                            ].get("type", "")

                    comfy_url_and_port_selector.change(
                        select_correct_port,
                        inputs=[comfy_url_and_port_selector],
                    )

                    run_button.click(
                        fn=run_workflow_with_name(
                            workflow_filename, workflow_name, components, component_dict
                        ),
                        inputs=components,
                        outputs=gr.Text(visible=False),
                        trigger_mode="multiple",
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
