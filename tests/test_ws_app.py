import gradio as gr
import json
import time
import threading
from typing import Dict, Set, Optional
import uuid
import asyncio
import websocket
import urllib.request
import urllib.parse
from PIL import Image
import io
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestWSApp")


class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class AppState:
    job_tracking: Dict = None
    current_progress_data: Dict = None
    websocket_manager = None

    def __post_init__(self):
        if self.job_tracking is None:
            self.job_tracking = {}
        if self.current_progress_data is None:
            self.current_progress_data = {}


class ComfyUIWebSocket:
    def __init__(self, server_address="127.0.0.1:8188", ws_address="127.0.0.1:8189"):
        self.server_address = server_address
        self.ws_address = ws_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.job_tracking: Dict[str, dict] = {}
        logger.info(
            f"Initializing ComfyUIWebSocket with server_address={server_address}, ws_address={ws_address}"
        )

    def connect(self):
        logger.info(
            f"Connecting to WebSocket at ws://{self.ws_address}/ws?clientId={self.client_id}"
        )
        if self.ws:
            logger.info("Closing existing WebSocket connection")
            self.ws.close()
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.ws_address}/ws?clientId={self.client_id}")
        logger.info("WebSocket connected successfully")

    def queue_prompt(self, prompt):
        logger.info("Queueing prompt...")
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        response = json.loads(urllib.request.urlopen(req).read())
        logger.info(f"Prompt queued successfully. Prompt ID: {response['prompt_id']}")
        return response

    def get_image(self, filename, subfolder, folder_type):
        logger.info(f"Fetching image: {filename} from {subfolder}/{folder_type}")
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            f"http://{self.server_address}/view?{url_values}"
        ) as response:
            image_data = response.read()
            logger.info(f"Image fetched successfully. Size: {len(image_data)} bytes")
            return image_data

    def get_history(self, prompt_id):
        logger.info(f"Fetching history for prompt ID: {prompt_id}")
        with urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        ) as response:
            history = json.loads(response.read())
            logger.info("History fetched successfully")
            return history

    def track_job(self, job_id: str):
        """Track a new generation job"""
        self.job_tracking[job_id] = {
            "status": "pending",
            "timestamp": time.time(),
            "output_file": None,
        }
        logger.info(f"Started tracking job {job_id}")

    def get_job_status(self, job_id: str):
        """Get the current status of a job"""
        if job_id not in self.job_tracking:
            return {"status": "not_found"}
        return self.job_tracking[job_id]

    def check_for_new_content(self):
        """Check for completed jobs and return their results"""
        completed_jobs = {
            job_id: job_data
            for job_id, job_data in self.job_tracking.items()
            if job_data["status"] == "completed"
        }
        return completed_jobs

    def generate_image(self, prompt_text, positive_prompt, negative_prompt, seed):
        logger.info(f"Starting image generation with seed={seed}")
        logger.info(f"Positive prompt: {positive_prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")

        try:
            # Parse the base prompt
            prompt = json.loads(prompt_text)

            # Update the prompts and seed
            prompt["6"]["inputs"]["text"] = positive_prompt
            prompt["7"]["inputs"]["text"] = negative_prompt
            prompt["3"]["inputs"]["seed"] = seed
            logger.info("Prompt template updated with new values")

            # Ensure we have a connection
            self.connect()

            # Queue the prompt and get the prompt ID
            response = self.queue_prompt(prompt)
            prompt_id = response["prompt_id"]
            logger.info(f"Waiting for execution to complete for prompt ID: {prompt_id}")

            # Track the job
            self.track_job(prompt_id)

            # Wait for execution to complete
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message["type"] == "executing":
                        data = message["data"]
                        logger.info(
                            f"Execution status: Node {data['node']} for prompt {data['prompt_id']}"
                        )
                        if data["node"] is None and data["prompt_id"] == prompt_id:
                            logger.info("Execution completed")
                            break

            # Get the history and images
            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    for image in node_output["images"]:
                        logger.info(f"Processing image from node {node_id}")
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        # Convert bytes to PIL Image
                        pil_image = Image.open(io.BytesIO(image_data))
                        # Update job status
                        self.job_tracking[prompt_id].update(
                            {"status": "completed", "output_file": image["filename"]}
                        )
                        logger.info("Image generation completed successfully")
                        return pil_image

        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}", exc_info=True)
            if prompt_id in self.job_tracking:
                self.job_tracking[prompt_id]["status"] = "failed"
            return f"Error: {str(e)}"
        finally:
            if self.ws:
                logger.info("Closing WebSocket connection")
                self.ws.close()


def create_test_app():
    logger.info("Creating test app")
    app_state = AppState()
    comfy = ComfyUIWebSocket()
    app_state.websocket_manager = comfy

    # Default prompt template
    default_prompt = """
    {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 8,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 8566257,
                "steps": 20
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": 512,
                "width": 512
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": "masterpiece best quality bavis"
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": "bad hands"
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            }
        }
    }
    """

    def generate(positive_prompt, negative_prompt, seed):
        logger.info("Generate function called from Gradio interface")
        result = comfy.generate_image(
            default_prompt, positive_prompt, negative_prompt, seed
        )
        # If result is a PIL Image, save it and return the path
        if isinstance(result, Image.Image):
            import os

            # Create a test directory in our project
            test_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
            os.makedirs(test_dir, exist_ok=True)

            # Save the image with a unique name
            filename = f"image_{uuid.uuid4()}.webp"
            filepath = os.path.join(test_dir, filename)
            result.save(filepath, format="WEBP")

            # Track this job with the filepath
            logger.info(f"Started tracking job {filepath}")
            comfy.track_job(filepath)
            comfy.job_tracking[filepath].update(
                {
                    "status": "completed",
                    "output_file": filepath,
                    "timestamp": time.time(),
                }
            )
            return filepath
        return result

    def translate_gradio_path(gradio_path):
        """Translate a Gradio temp path to our test directory path"""
        import os

        if not gradio_path:
            return None

        # Extract the filename from the Gradio path
        filename = os.path.basename(gradio_path)
        if not filename.startswith("image_"):
            return None

        # Create the corresponding path in our test directory
        test_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        return os.path.join(test_dir, filename)

    def get_client_id():
        """Get a new client ID for WebSocket connection"""
        return str(uuid.uuid4())

    def get_job_result(job_id: str):
        """Get the current status and result for a job"""
        logger.info(f"Checking job status for: {job_id}")

        # Try to translate the Gradio path to our test directory path
        translated_path = translate_gradio_path(job_id)
        if translated_path:
            logger.info(f"Translated path: {translated_path}")
            job_id = translated_path

        if job_id in comfy.job_tracking:
            job = comfy.job_tracking[job_id]
            logger.info(f"Found job with status: {job['status']}")
            return {
                "status": job["status"],
                "output_file": job["output_file"],
                "timestamp": job["timestamp"],
            }
        logger.info(f"Job not found: {job_id}")
        return {"status": "not_found"}

    def check_for_new_content():
        """Check for completed jobs and return their results"""
        completed_jobs = comfy.check_for_new_content()
        logger.info(
            f"Checking for new content. Found {len(completed_jobs)} completed jobs"
        )
        # If no completed jobs, return an empty dict
        if not completed_jobs:
            return {}
        # Return the first completed job
        job_id = next(iter(completed_jobs))
        logger.info(f"Returning completed job: {job_id}")
        return {job_id: completed_jobs[job_id]}

    # Create the Gradio interface
    logger.info("Creating Gradio interface")
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                positive_prompt = gr.Textbox(
                    label="Positive Prompt", value="masterpiece best quality bavis"
                )
                negative_prompt = gr.Textbox(label="Negative Prompt", value="bad hands")
                seed = gr.Number(label="Seed", value=5)
                generate_button = gr.Button("Generate Image")
                output_image = gr.Image(label="Generated Image")

        # Add API endpoints
        generate_button.click(
            fn=generate,
            inputs=[positive_prompt, negative_prompt, seed],
            outputs=output_image,
            api_name="workflow/test",
        )

        # Add hidden components for API endpoints
        client_id_input = gr.Text(visible=False)
        client_id_output = gr.Text(visible=False)
        client_id_input.submit(
            fn=get_client_id,
            inputs=[],
            outputs=client_id_output,
            api_name="get_client_id",
        )

        result_input = gr.Text(visible=False)
        result_output = gr.JSON(visible=False)
        result_input.submit(
            fn=get_job_result,
            inputs=result_input,
            outputs=result_output,
            api_name="workflow_result",
        )

        # Add check_for_new_content endpoint
        check_content_input = gr.Text(visible=False)
        check_content_output = gr.JSON(visible=False)
        check_content_input.submit(
            fn=check_for_new_content,
            inputs=[],
            outputs=check_content_output,
            api_name="check_for_new_content",
        )

    # Add cleanup on close
    logger.info("Launching Gradio interface")
    iface.launch(show_error=True)

    return iface


if __name__ == "__main__":
    logger.info("Starting test app")
    app = create_test_app()
