import gradio as gr
import json
import time
import threading
from typing import Dict, Set
import uuid
import asyncio
import websockets
from websockets.server import WebSocketServerProtocol

# Store active jobs and their status
jobs: Dict[str, dict] = {}

# Store active WebSocket connections
active_connections: Set[WebSocketServerProtocol] = set()

def start_job():
    """Simulate starting a job that will complete after some time"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "output_file": None
    }
    
    # Start a thread to simulate job completion
    def complete_job():
        time.sleep(5)  # Simulate work
        jobs[job_id] = {
            "status": "completed",
            "output_file": f"output_{job_id}.png"
        }
        # Notify all connected clients about job completion
        asyncio.run(notify_job_completion(job_id))
    
    threading.Thread(target=complete_job, daemon=True).start()
    return job_id

def get_job_status(job_id: str):
    """Get the current status of a job"""
    if job_id not in jobs:
        return {"status": "not_found"}
    return jobs[job_id]

def check_for_new_content():
    """Check for completed jobs and return their results"""
    completed_jobs = {
        job_id: job_data 
        for job_id, job_data in jobs.items() 
        if job_data["status"] == "completed"
    }
    return completed_jobs

async def handle_websocket_message(ws: WebSocketServerProtocol, message: str):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        if data.get("type") == "subscribe_job":
            job_id = data.get("job_id")
            if job_id in jobs:
                # Send current job status
                response = {
                    "type": "job_update",
                    "job_id": job_id,
                    "status": jobs[job_id]["status"],
                    "output_file": jobs[job_id]["output_file"]
                }
                await ws.send(json.dumps(response))
    except Exception as e:
        print(f"Error handling WebSocket message: {e}")

async def notify_job_completion(job_id: str):
    """Notify all connected clients about job completion"""
    if job_id in jobs:
        job = jobs[job_id]
        message = {
            "type": "job_update",
            "job_id": job_id,
            "status": job["status"],
            "output_file": job["output_file"]
        }
        message_str = json.dumps(message)
        
        # Send to all active connections
        for ws in list(active_connections):
            try:
                await ws.send(message_str)
            except Exception as e:
                print(f"Error sending to WebSocket: {e}")
                active_connections.discard(ws)

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            start_button = gr.Button("Start Test Job")
            job_id_output = gr.Textbox(label="Job ID")
            
            # Hidden components for the result endpoint
            result_input = gr.Text(visible=False)
            result_output = gr.JSON(visible=False)
            
            # Add the result endpoint
            result_input.submit(
                fn=get_job_status,
                inputs=result_input,
                outputs=result_output,
                api_name="workflow_result"
            )
            
        with gr.Column():
            status_output = gr.JSON(label="Job Status")
    
    # Set up the start button
    start_button.click(
        fn=start_job,
        outputs=job_id_output
    )
    
    # Set up polling for job status
    timer = gr.Timer(value=1.0)
    timer.tick(
        fn=check_for_new_content,
        outputs=status_output
    )

if __name__ == "__main__":
    async def start_websocket_server():
        async def handler(websocket: WebSocketServerProtocol):
            active_connections.add(websocket)
            try:
                async for message in websocket:
                    if message:
                        await handle_websocket_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
            except Exception as e:
                print(f"Error in WebSocket handler: {e}")
            finally:
                active_connections.discard(websocket)
        
        server = await websockets.serve(handler, "localhost", 8765)
        print("WebSocket server started on ws://localhost:8765")
        await server.wait_closed()
    
    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(
        target=lambda: asyncio.run(start_websocket_server()),
        daemon=True
    )
    websocket_thread.start()
    
    # Launch Gradio app
    demo.launch() 