import asyncio
from gradio_client import Client
import time
import json
import websockets
from typing import Dict, Optional
from websockets.client import WebSocketClientProtocol

class TestClient:
    def __init__(self, host: str = "http://localhost:7860"):
        self.client = Client(host)
        self.ws: Optional[WebSocketClientProtocol] = None
        self.job_futures: Dict[str, asyncio.Future] = {}
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._connect_websocket())
        
    async def _connect_websocket(self):
        """Connect to the WebSocket server"""
        try:
            ws_url = "ws://localhost:8765"  # WebSocket server from test_ws_app.py
            self.ws = await websockets.connect(ws_url)
            print(f"Connected to WebSocket at {ws_url}")
            
            # Start WebSocket message handling
            self.loop.create_task(self._handle_messages())
            
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            self.ws = None

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        if not self.ws:
            return
            
        try:
            async for message in self.ws:
                if message:
                    await self._handle_ws_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

    async def _handle_ws_message(self, message: str):
        """Process a WebSocket message"""
        try:
            data = json.loads(message)
            if data.get("type") == "job_update":
                job_id = data.get("job_id")
                if job_id in self.job_futures:
                    future = self.job_futures[job_id]
                    if not future.done():
                        if data["status"] == "completed":
                            future.set_result(data["output_file"])
                        elif data["status"] == "failed":
                            future.set_exception(Exception(f"Job {job_id} failed"))
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
        
    async def _poll_until_complete(self, job_id: str) -> str:
        """Poll for job completion using WebSocket with HTTP fallback"""
        print(f"Starting job tracking for {job_id}")
        
        # Create a future for this job
        future = asyncio.Future()
        self.job_futures[job_id] = future
        
        try:
            # Subscribe to job updates via WebSocket
            if self.ws and self.ws.open:
                try:
                    subscribe_msg = {
                        "type": "subscribe_job",
                        "job_id": job_id
                    }
                    await self.ws.send(json.dumps(subscribe_msg))
                    print(f"Subscribed to job {job_id} updates via WebSocket")
                except Exception as e:
                    print(f"Failed to subscribe via WebSocket: {e}")
            
            # Fallback to HTTP polling
            attempts = 0
            while not future.done():
                try:
                    result = self.client.predict(job_id, api_name="/workflow_result")
                    print(f"Polling job {job_id}: {result}")
                    
                    if result["status"] == "completed":
                        if result.get("output_file"):
                            future.set_result(result["output_file"])
                            break
                        raise Exception(f"Job {job_id} completed but no output file found")
                    elif result["status"] == "failed":
                        future.set_exception(Exception(f"Job {job_id} failed: {result.get('error', 'Unknown error')}"))
                        break
                        
                except Exception as e:
                    print(f"Error polling job {job_id}: {str(e)}")
                
                attempts += 1
                sleep_time = min(2.0 * (1.2 ** (attempts - 1)), 5.0)
                await asyncio.sleep(sleep_time)
            
            # Wait for the future to complete
            return await future
            
        finally:
            # Clean up the future
            if job_id in self.job_futures:
                del self.job_futures[job_id]
            
    async def start_test_job(self):
        """Start a test job and wait for completion"""
        try:
            # Start the job
            job_id = self.client.predict(api_name="/start_job")
            print(f"Started job with ID: {job_id}")
            
            # Wait for completion
            result = await self._poll_until_complete(job_id)
            print(f"Job completed with result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in start_test_job: {e}")
            raise

async def main():
    client = TestClient()
    try:
        result = await client.start_test_job()
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 