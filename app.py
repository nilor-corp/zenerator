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
from threading import Timer
import websockets


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
ZENERATOR_IP = config["ZENERATOR_IP"]
ZENERATOR_PORT = config["ZENERATOR_PORT"]
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


# # REF: https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/blob/master/export_trt.py
# def install_tensorrt():
#     print("Installing TensorRT...")

#     engine = Engine(UPSCALER_DIR + "\\realistic.engine")

#     torch.cuda.empty_cache()

#     s = time.time()
#     ret = engine.build(
#         ONNX_PATH,
#         True,
#         enable_preview=True,
#         input_profile=[
#             {
#                 "input": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 1280, 1280)]
#             },  # any sizes from 256x256 to 1280x1280
#         ],
#     )
#     e = time.time()
#     print(f"Time taken to build TensorRT: {(e-s)} seconds")

#     return ret


# if os.path.exists(TENSORRT_NODES_DIR):
#     print(f"Importing TensorRT requirements.")

#     sys.path.append(TENSORRT_NODES_DIR)

#     # from export_trt import export_trt
#     from utilities import Engine

# try:
#     if check_tensorrt_installation():
#         print(f"TensorRT is already installed.")
#     else:
#         install_tensorrt()
# except Exception as e:
#     print(f"Error installing TensorRT: {e}")


# endregion
# region Websocket
class WebSocketManager:
    def __init__(self):
        self.ws = None
        self.client_id = str(uuid.uuid4())
        self.clients = {}  # Store client connections
        self.job_tracking = {}  # Track jobs for each client

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

    async def handle_ws_message(self, message: str, websocket):
        """Handle incoming WebSocket messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "submit_workflow":
                # Handle workflow submission
                workflow_type = data["data"]["workflow_type"]
                params = data["data"]["params"]
                client_id = data["data"]["client_id"]
                
                # Run the workflow
                job_id = run_workflow(
                    workflow_definitions[workflow_type]["filename"],
                    workflow_type,
                    None,  # progress parameter not needed for WebSocket
                    **params
                )
                
                # Send confirmation
                response = {
                    "type": "workflow_submitted",
                    "data": {
                        "job_id": job_id,
                        "client_id": client_id
                    }
                }
                websocket.send(json.dumps(response))
                
            elif message_type == "get_job_result":
                # Handle job result request
                job_id = data["data"]["job_id"]
                client_id = data["data"]["client_id"]
                
                # Get job status
                result = get_job_result(job_id)
                
                # Send response
                response = {
                    "type": "job_result",
                    "data": {
                        "job_id": job_id,
                        "client_id": client_id,
                        **result
                    }
                }
                websocket.send(json.dumps(response))
                
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
            # Send error response
            error_response = {
                "type": "error",
                "data": {
                    "error": str(e)
                }
            }
            websocket.send(json.dumps(error_response))

    async def broadcast_execution_status(self, prompt_id: str):
        """Broadcast execution status to all connected clients"""
        try:
            # Get job status
            result = get_job_result(prompt_id)
            
            # Prepare message
            message = {
                "type": "executing",
                "data": {
                    "prompt_id": prompt_id,
                    "status": result["status"],
                    "output_file": result.get("output_file")
                }
            }
            
            # Broadcast to all clients
            for client_id, client_ws in self.clients.items():
                try:
                    client_ws.send(json.dumps(message))
                except Exception as e:
                    print(f"Error sending to client {client_id}: {e}")
                    
        except Exception as e:
            print(f"Error broadcasting execution status: {e}")

    def cleanup(self):
        """Clean up WebSocket connections"""
        for client_id, client_ws in self.clients.items():
            try:
                client_ws.close()
            except Exception as e:
                print(f"Error closing client {client_id} connection: {e}")
        self.clients.clear()

# Update the WebSocket server setup in the main app
def create_websocket_server():
    """Create and start the WebSocket server"""
    async def websocket_handler(websocket, path):
        client_id = str(uuid.uuid4())
        websocket_manager.clients[client_id] = websocket
        try:
            async for message in websocket:
                await websocket_manager.handle_ws_message(message, websocket)
        finally:
            del websocket_manager.clients[client_id]

    return websockets.serve(websocket_handler, "localhost", 8189)

# Update the load_demo function to start the WebSocket server
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

            # Start WebSocket server
            ws_server = create_websocket_server()
            asyncio.run(ws_server)

        except Exception as e:
            print(f"An error occurred: {type(e).__name__}: {e}")

# Update the unload_demo function to clean up WebSocket connections
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
    tick_timer = None

    # Clean up WebSocket connections
    app_state.websocket_manager.cleanup()

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
    print("\nCreating tabs...")
    tick_timer = gr.Timer(value=1.0)
    demo.load(fn=load_demo)
    demo.unload(fn=unload_demo)

    with gr.Row():
        with gr.Column(scale=5):
            tabs = create_tabs()

        with gr.Column(scale=4):
            # Right column content remains the same
            gr.Markdown("### Output Preview")
            with gr.Group():
                if output_type == "image":
                    latest_content = get_latest_image(OUT_DIR)
                    if latest_content is not None:
                        output_player = gr.Image(
                            value=latest_content,
                            show_label=False,
                            interactive=False,
                        )
                    else:
                        output_player = gr.Image(
                            show_label=False,
                            interactive=False,
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
                fn=check_interrupt_visibility,
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
                interrupt_button = gr.Button("Interrupt", variant="stop", visible=True)
                interrupt_button.click(fn=post_interrupt)

            tick_timer.tick(
                fn=check_interrupt_visibility,
                outputs=interrupt_group,
                show_progress="hidden",
            )

    # Create hidden components for the result endpoint
    result_input = gr.Text(visible=False)
    result_output = gr.JSON(visible=False)
    result_input.submit(
        fn=get_job_result,
        inputs=result_input,
        outputs=result_output,
        api_name="workflow_result",
    )

    if __name__ == "__main__":
        setup_signal_handlers()
        initialize_content_tracking()
        demo.queue()

        demo.launch(
            allowed_paths=allowed_paths,
            favicon_path="favicon.png",
            server_name=ZENERATOR_IP,
            server_port=ZENERATOR_PORT,
            inbrowser=True,
        )

def create_tabs():
    """Create and return the main interface tabs"""
    global _tabs_created

    # Check if tabs have already been created
    if _tabs_created:
        print("Skipping tab creation (already created)")
        return gr.Tabs()

    _tabs_created = True  # Set flag before creating to prevent recursion
    print("Creating About tab")
    tabs = gr.Tabs()
    with tabs:
        with gr.TabItem(label="About"):
            gr.Markdown(
                "Zenerator is a tool by Nilor Corp for creating and running generative AI workflows.\n\n"
                "Select a workflow from the tabs above and fill in the parameters.\n\n"
                "Click 'Run Workflow' to start the workflow.",
                line_breaks=True,
            )

        print(f"\nWorkflow definitions keys: {list(workflow_definitions.keys())}")
        for workflow_name in workflow_definitions.keys():
            workflow_filename = workflow_definitions[workflow_name]["filename"]
            print(f"\nCreating tab for workflow: {workflow_name}")
            print(f"Workflow name: {workflow_definitions[workflow_name]['name']}")

            with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):
                print(f"Creating content for {workflow_name}")
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

                        components, component_dict = create_tab_interface(workflow_name)

                    output_type = workflow_definitions[workflow_name]["outputs"].get(
                        "type", ""
                    )

                comfy_url_and_port_selector.change(
                    select_correct_port,
                    inputs=[comfy_url_and_port_selector],
                )

                run_button.click(
                    fn=run_workflow_with_name(
                        workflow_filename,
                        workflow_name,
                        components,
                        component_dict,
                    ),
                    inputs=components,
                    outputs=gr.Text(visible=False),
                    trigger_mode="multiple",
                    api_name=f"workflow/{workflow_name}",
                )
                print(f"Finished creating tab for {workflow_name}")

    return tabs
