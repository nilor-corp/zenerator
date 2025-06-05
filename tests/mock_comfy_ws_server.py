import json
import threading
import time
import websockets
import asyncio
import base64
from PIL import Image
import io
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MockComfyUI')

class MockComfyUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(f"HTTP {format%args}")

    def do_POST(self):
        if self.path == '/prompt':
            logger.info("Received POST request to /prompt")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Generate a fake prompt ID
            prompt_id = str(uuid.uuid4())
            logger.info(f"Generated prompt ID: {prompt_id}")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'prompt_id': prompt_id}
            self.wfile.write(json.dumps(response).encode())
            logger.info("Sent prompt response")
            
            # Start a background thread to simulate processing
            logger.info("Starting processing simulation thread")
            # Get the server instance to access the broadcast method
            server = self.server.server_instance
            threading.Thread(target=server.simulate_processing, args=(prompt_id,)).start()
    
    def do_GET(self):
        if self.path.startswith('/history/'):
            prompt_id = self.path.split('/')[-1]
            logger.info(f"Received GET request for history of prompt ID: {prompt_id}")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Create fake history response
            history = {
                prompt_id: {
                    'outputs': {
                        '9': {  # Node ID for SaveImage
                            'images': [{
                                'filename': 'fake_image.png',
                                'subfolder': 'ComfyUI',
                                'type': 'output'
                            }]
                        }
                    }
                }
            }
            self.wfile.write(json.dumps(history).encode())
            logger.info("Sent history response")
        
        elif self.path.startswith('/view'):
            logger.info("Received GET request for image view")
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = dict(urllib.parse.parse_qsl(query))
            logger.info(f"Image request parameters: {params}")
            
            # Create a fake image
            img = Image.new('RGB', (512, 512), color='white')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(img_byte_arr)
            logger.info(f"Sent image response. Size: {len(img_byte_arr)} bytes")

class MockComfyUIWebSocketServer:
    def __init__(self, host='localhost', http_port=8188, ws_port=8189):
        logger.info(f"Initializing MockComfyUIWebSocketServer with host={host}, http_port={http_port}, ws_port={ws_port}")
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.clients = set()
        self.http_server = None
        self.ws_server = None
    
    async def websocket_handler(self, websocket):
        client_id = str(uuid.uuid4())
        logger.info(f"New WebSocket client connected. Client ID: {client_id}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                logger.debug(f"Received WebSocket message from client {client_id}: {message}")
        finally:
            logger.info(f"WebSocket client disconnected. Client ID: {client_id}")
            self.clients.remove(websocket)
    
    async def broadcast_execution_status(self, prompt_id):
        logger.info(f"Broadcasting execution status for prompt ID: {prompt_id}")
        # Simulate execution status updates
        status_messages = [
            {'type': 'executing', 'data': {'node': '4', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '5', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '6', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '7', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '3', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '8', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '9', 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': None, 'prompt_id': prompt_id}}
        ]
        
        for message in status_messages:
            logger.info(f"Broadcasting status: {message}")
            for client in self.clients:
                await client.send(json.dumps(message))
            await asyncio.sleep(0.5)
        logger.info("Execution status broadcast completed")
    
    def simulate_processing(self, prompt_id):
        logger.info(f"Starting processing simulation for prompt ID: {prompt_id}")
        # Simulate processing time
        time.sleep(1)
        logger.info(f"Processing simulation completed for prompt ID: {prompt_id}")
        
        # Send execution status through WebSocket
        asyncio.run(self.broadcast_execution_status(prompt_id))
    
    def start(self):
        logger.info("Starting mock server...")
        # Start HTTP server
        self.http_server = HTTPServer((self.host, self.http_port), MockComfyUIHandler)
        self.http_server.server_instance = self  # Store reference to self for handler access
        http_thread = threading.Thread(target=self.http_server.serve_forever)
        http_thread.daemon = True
        http_thread.start()
        logger.info(f"HTTP server started on port {self.http_port}")
        
        # Start WebSocket server
        async def start_ws_server():
            self.ws_server = await websockets.serve(
                self.websocket_handler,
                self.host,
                self.ws_port
            )
            logger.info(f"WebSocket server started on port {self.ws_port}")
            await self.ws_server.wait_closed()
        
        ws_thread = threading.Thread(target=lambda: asyncio.run(start_ws_server()))
        ws_thread.daemon = True
        ws_thread.start()
        logger.info("Mock server started successfully")
    
    def stop(self):
        logger.info("Stopping mock server...")
        if self.http_server:
            logger.info("Shutting down HTTP server...")
            self.http_server.shutdown()
            logger.info("HTTP server stopped")
        if self.ws_server:
            logger.info("Closing WebSocket server...")
            asyncio.run(self.ws_server.close())
            logger.info("WebSocket server stopped")
        logger.info("Mock server stopped successfully")

def create_mock_server():
    logger.info("Creating new mock server instance")
    server = MockComfyUIWebSocketServer()
    server.start()
    return server

if __name__ == "__main__":
    logger.info("Starting mock server in standalone mode")
    server = create_mock_server()
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        server.stop()
