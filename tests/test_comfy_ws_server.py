import gradio as gr
import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import time
import logging
from mock_comfy_ws_server import create_mock_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ComfyUIWebSocket')

class ComfyUIWebSocket:
    def __init__(self, server_address="127.0.0.1:8188", use_mock=True):
        self.server_address = server_address
        self.ws_address = "127.0.0.1:8189" if use_mock else server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.mock_server = None
        
        logger.info(f"Initializing ComfyUIWebSocket with server_address={server_address}, ws_address={self.ws_address}, use_mock={use_mock}")
        
        if use_mock:
            logger.info("Starting mock server...")
            self.mock_server = create_mock_server()
            logger.info("Mock server started successfully")

    def connect(self):
        logger.info(f"Connecting to WebSocket at ws://{self.ws_address}/ws?clientId={self.client_id}")
        if self.ws:
            logger.info("Closing existing WebSocket connection")
            self.ws.close()
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.ws_address}/ws?clientId={self.client_id}")
        logger.info("WebSocket connected successfully")

    def queue_prompt(self, prompt):
        logger.info("Queueing prompt...")
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        response = json.loads(urllib.request.urlopen(req).read())
        logger.info(f"Prompt queued successfully. Prompt ID: {response['prompt_id']}")
        return response

    def get_image(self, filename, subfolder, folder_type):
        logger.info(f"Fetching image: {filename} from {subfolder}/{folder_type}")
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            image_data = response.read()
            logger.info(f"Image fetched successfully. Size: {len(image_data)} bytes")
            return image_data

    def get_history(self, prompt_id):
        logger.info(f"Fetching history for prompt ID: {prompt_id}")
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            history = json.loads(response.read())
            logger.info("History fetched successfully")
            return history

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
            prompt_id = self.queue_prompt(prompt)['prompt_id']
            logger.info(f"Waiting for execution to complete for prompt ID: {prompt_id}")
            output_images = {}

            # Wait for execution to complete
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        logger.info(f"Execution status: Node {data['node']} for prompt {data['prompt_id']}")
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            logger.info("Execution completed")
                            break

            # Get the history and images
            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        logger.info(f"Processing image from node {node_id}")
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        # Convert bytes to PIL Image
                        pil_image = Image.open(io.BytesIO(image_data))
                        logger.info("Image generation completed successfully")
                        return pil_image

        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
        finally:
            if self.ws:
                logger.info("Closing WebSocket connection")
                self.ws.close()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        if self.mock_server:
            logger.info("Stopping mock server...")
            self.mock_server.stop()
            logger.info("Mock server stopped")
        if self.ws:
            logger.info("Closing WebSocket connection...")
            self.ws.close()
            logger.info("WebSocket connection closed")

def create_test_app(use_mock=True):
    logger.info(f"Creating test app with use_mock={use_mock}")
    comfy = ComfyUIWebSocket(use_mock=use_mock)
    
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
        return comfy.generate_image(default_prompt, positive_prompt, negative_prompt, seed)

    # Create the Gradio interface
    logger.info("Creating Gradio interface")
    iface = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Positive Prompt", value="masterpiece best quality bavis"),
            gr.Textbox(label="Negative Prompt", value="bad hands"),
            gr.Number(label="Seed", value=5)
        ],
        outputs=gr.Image(label="Generated Image"),
        title="ComfyUI WebSocket Test",
        description="Test the ComfyUI WebSocket API with a simple image generation interface"
    )
    
    # Add cleanup on close
    logger.info("Launching Gradio interface")
    iface.launch(show_error=True)
    comfy.cleanup()
    
    return iface

if __name__ == "__main__":
    logger.info("Starting test app with mock server")
    app = create_test_app(use_mock=True)  # Use mock server by default
