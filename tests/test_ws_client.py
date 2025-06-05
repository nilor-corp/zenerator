import asyncio
from gradio_client import Client
import time
import json
import websocket
from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestWSClient')

class WorkflowType(Enum):
    TEST = "test"

@dataclass
class TestParams:
    width: int = 1024
    height: int = 1024
    prefix: str = "Zenerator/test"
    prompt: str = ""
    negative_prompt: str = ""

class TestClient:
    def __init__(self, host: str = "http://localhost:7860"):
        self.client = Client(host)
        self.ws: Optional[websocket.WebSocket] = None
        self.job_futures: Dict[str, asyncio.Future] = {}
        self.loop = asyncio.get_event_loop()
        self.client_id = None
        self.connected = False
        logger.info(f"Initializing TestClient with host={host}")

    def connect(self):
        """Connect to the WebSocket server"""
        try:
            # Connect directly to WebSocket since we don't need a client ID
            ws_url = f"ws://localhost:8189/ws"
            logger.info(f"Connecting to WebSocket at {ws_url}")
            
            self.ws = websocket.WebSocket()
            self.ws.connect(ws_url)
            self.connected = True
            logger.info("WebSocket connected successfully")
            
            # Start WebSocket message handling
            self.loop.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.ws = None
            self.connected = False

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        if not self.ws:
            return
            
        try:
            while self.connected:
                try:
                    message = self.ws.recv()
                    if message:
                        await self._handle_ws_message(message)
                except websocket.WebSocketConnectionClosedException:
                    logger.info("WebSocket connection closed")
                    self.connected = False
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break
        finally:
            self.connected = False
            if self.ws:
                self.ws.close()

    async def _handle_ws_message(self, message: str):
        """Process a WebSocket message"""
        try:
            data = json.loads(message)
            logger.info(f"Received message: {data}")
            
            if data.get("type") == "executing":
                prompt_id = data["data"]["prompt_id"]
                if prompt_id in self.job_futures:
                    future = self.job_futures[prompt_id]
                    if not future.done():
                        if data["data"]["node"] is None:  # Execution completed
                            # Get the result from the HTTP endpoint
                            result = await self._get_job_result(prompt_id)
                            if result["status"] == "completed":
                                future.set_result(result["output_file"])
                            else:
                                future.set_exception(Exception(f"Job {prompt_id} failed"))
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result from HTTP endpoint"""
        try:
            # Call the workflow_result endpoint with explicit api_name
            result = self.client.predict(
                job_id,
                api_name="/workflow_result"  # Add leading slash
            )
            logger.info(f"Got job result for {job_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting job result: {e}")
            return {"status": "failed", "error": str(e)}

    async def _poll_until_complete(self, job_id: str) -> str:
        """Poll for job completion using WebSocket with HTTP fallback"""
        logger.info(f"Starting job tracking for {job_id}")
        
        # Create a future for this job
        future = asyncio.Future()
        self.job_futures[job_id] = future
        
        try:
            # Ensure we have a WebSocket connection
            if not self.connected:
                self.connect()
            
            # Wait for the future to complete
            return await future
            
        finally:
            # Clean up the future
            if job_id in self.job_futures:
                del self.job_futures[job_id]
            
    async def generate_test(self, params: TestParams) -> str:
        """Generate image using test workflow"""
        try:
            # Submit the workflow with explicit api_name
            result = self.client.predict(
                params.prompt,
                params.negative_prompt,
                5,  # seed
                api_name="/workflow/test"  # Add leading slash
            )

            logger.info(f"Workflow submitted. Response: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_test: {e}")
            raise

async def main():
    client = TestClient()
    try:
        # Test parameters
        params = TestParams(
            prompt="masterpiece best quality bavis",
            negative_prompt="bad hands"
        )
        
        # Generate test image
        result = await client.generate_test(params)
        logger.info(f"Final result: {result}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        if client.ws:
            client.ws.close()

if __name__ == "__main__":
    asyncio.run(main()) 