import json
import threading
import time
import asyncio
import uuid
import logging
from PIL import Image
import io
import urllib.parse
from aiohttp import web, WSMsgType
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MockComfyUI')

class MockComfyUIWebSocketServer:
    def __init__(self, host='localhost', port=8188):
        logger.info(f"Initializing MockComfyUIWebSocketServer with host={host}, port={port}")
        self.host = host
        self.port = port
        self.clients = set()
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Set up routes
        self.setup_routes()
    
    def setup_routes(self):
        """Set up HTTP and WebSocket routes"""
        
        # HTTP routes
        self.app.router.add_post('/prompt', self.handle_prompt)
        self.app.router.add_post('/interrupt', self.handle_interrupt)
        self.app.router.add_post('/history', self.handle_history_post)
        self.app.router.add_post('/free', self.handle_free)
        
        self.app.router.add_get('/queue', self.handle_queue)
        self.app.router.add_get('/history', self.handle_history_get)
        self.app.router.add_get('/system_stats', self.handle_system_stats)
        self.app.router.add_get('/prompt', self.handle_prompt_get)
        self.app.router.add_get('/history/{prompt_id}', self.handle_history_specific)
        self.app.router.add_get('/view', self.handle_view)
        
        # WebSocket route
        self.app.router.add_get('/ws', self.handle_websocket)
    
    async def handle_prompt(self, request):
        """Handle POST /prompt"""
        logger.info("Received POST request to /prompt")
        data = await request.json()
        
        # Generate a fake prompt ID
        prompt_id = str(uuid.uuid4())
        logger.info(f"Generated prompt ID: {prompt_id}")
        
        # Start a background thread to simulate processing
        logger.info("Starting processing simulation thread")
        threading.Thread(target=self.simulate_processing, args=(prompt_id,)).start()
        
        return web.json_response({'prompt_id': prompt_id})
    
    async def handle_interrupt(self, request):
        """Handle POST /interrupt"""
        logger.info("Handling /interrupt request")
        return web.json_response({'success': True})
    
    async def handle_history_post(self, request):
        """Handle POST /history"""
        logger.info("Handling /history POST request")
        data = await request.json()
        
        if 'clear' in data and data['clear']:
            logger.info("Clearing history")
            return web.json_response({'success': True, 'message': 'History cleared'})
        elif 'delete' in data:
            prompt_id = data['delete']
            logger.info(f"Deleting prompt ID: {prompt_id}")
            return web.json_response({'success': True, 'message': f'Prompt {prompt_id} deleted'})
        else:
            return web.json_response({'success': True})
    
    async def handle_free(self, request):
        """Handle POST /free"""
        logger.info("Handling /free request")
        data = await request.json()
        
        unload_models = data.get('unload_models', True)
        free_memory = data.get('free_memory', True)
        logger.info(f"Freeing memory - unload_models: {unload_models}, free_memory: {free_memory}")
        
        return web.json_response({'success': True, 'message': 'Memory freed'})
    
    async def handle_queue(self, request):
        """Handle GET /queue"""
        logger.info("Handling /queue request")
        queue_response = {
            'queue_running': [],
            'queue_pending': [],
            'queue_failed': []
        }
        return web.json_response(queue_response)
    
    async def handle_history_get(self, request):
        """Handle GET /history"""
        logger.info("Handling /history request")
        history_response = {}
        return web.json_response(history_response)
    
    async def handle_system_stats(self, request):
        """Handle GET /system_stats"""
        logger.info("Handling /system_stats request")
        system_stats_response = {
            'devices': [
                {
                    'vram_free': 8192,  # 8GB free
                    'vram_total': 12288,  # 12GB total
                    'name': 'NVIDIA GeForce RTX 3080'
                }
            ],
            'ram_free': 16384,
            'ram_total': 32768
        }
        return web.json_response(system_stats_response)
    
    async def handle_prompt_get(self, request):
        """Handle GET /prompt"""
        logger.info("Handling /prompt request")
        prompt_response = {
            'status': 'idle'
        }
        return web.json_response(prompt_response)
    
    async def handle_history_specific(self, request):
        """Handle GET /history/{prompt_id}"""
        prompt_id = request.match_info['prompt_id']
        logger.info(f"Received GET request for history of prompt ID: {prompt_id}")
        
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
        return web.json_response(history)
    
    async def handle_view(self, request):
        """Handle GET /view"""
        logger.info("Received GET request for image view")
        
        # Create a fake image
        img = Image.new('RGB', (512, 512), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return web.Response(body=img_byte_arr, content_type='image/png')
    
    async def handle_websocket(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = str(uuid.uuid4())
        logger.info(f"New WebSocket client connected. Client ID: {client_id}")
        self.clients.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    logger.debug(f"Received WebSocket message from client {client_id}: {msg.data}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error for client {client_id}: {ws.exception()}")
        except Exception as e:
            logger.error(f"Error in WebSocket handler for client {client_id}: {e}")
        finally:
            logger.info(f"WebSocket client disconnected. Client ID: {client_id}")
            self.clients.discard(ws)
        
        return ws
    
    async def broadcast_execution_status(self, prompt_id):
        logger.info(f"Broadcasting execution status for prompt ID: {prompt_id}")
        
        # Simulate execution status updates with more realistic progress
        execution_sequence = [
            {'type': 'executing', 'data': {'node': '4', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 1, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '5', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 5, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '6', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 10, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '7', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 15, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '3', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 18, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '8', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 19, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': '9', 'prompt_id': prompt_id}},
            {'type': 'progress', 'data': {'value': 20, 'max': 20, 'prompt_id': prompt_id}},
            {'type': 'executing', 'data': {'node': None, 'prompt_id': prompt_id}},
            {'type': 'status', 'data': {'status': {'exec_info': {'value': 20, 'max': 20, 'prompt_id': prompt_id}}}}
        ]
        
        for message in execution_sequence:
            logger.info(f"Broadcasting status: {message}")
            for client in self.clients:
                try:
                    await client.send_str(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to client: {e}")
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
        
        async def start_server():
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            logger.info(f"Server started on {self.host}:{self.port}")
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        
        # Start the server in a separate thread
        server_thread = threading.Thread(target=lambda: asyncio.run(start_server()))
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        logger.info("Mock server started successfully")
    
    def stop(self):
        logger.info("Stopping mock server...")
        if self.runner:
            logger.info("Shutting down server...")
            try:
                # Close all client connections
                for client in list(self.clients):
                    try:
                        asyncio.run(client.close())
                    except Exception as e:
                        logger.error(f"Error closing client connection: {e}")
                
                # Stop the server
                asyncio.run(self.runner.cleanup())
                logger.info("Server stopped")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
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
