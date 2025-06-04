import gradio as gr
import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import time

class ComfyUIWebSocket:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect(self):
        if self.ws:
            self.ws.close()
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def generate_image(self, prompt_text, positive_prompt, negative_prompt, seed):
        try:
            # Parse the base prompt
            prompt = json.loads(prompt_text)
            
            # Update the prompts and seed
            prompt["6"]["inputs"]["text"] = positive_prompt
            prompt["7"]["inputs"]["text"] = negative_prompt
            prompt["3"]["inputs"]["seed"] = seed

            # Ensure we have a connection
            self.connect()

            # Queue the prompt and get the prompt ID
            prompt_id = self.queue_prompt(prompt)['prompt_id']
            output_images = {}

            # Wait for execution to complete
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break

            # Get the history and images
            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        # Convert bytes to PIL Image
                        pil_image = Image.open(io.BytesIO(image_data))
                        return pil_image

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if self.ws:
                self.ws.close()

def create_test_app():
    comfy = ComfyUIWebSocket()
    
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
                "text": "masterpiece best quality girl"
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
        return comfy.generate_image(default_prompt, positive_prompt, negative_prompt, seed)

    # Create the Gradio interface
    iface = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Positive Prompt", value="masterpiece best quality man"),
            gr.Textbox(label="Negative Prompt", value="bad hands"),
            gr.Number(label="Seed", value=5)
        ],
        outputs=gr.Image(label="Generated Image"),
        title="ComfyUI WebSocket Test",
        description="Test the ComfyUI WebSocket API with a simple image generation interface"
    )
    
    return iface

if __name__ == "__main__":
    app = create_test_app()
    app.launch()
