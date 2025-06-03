import gradio as gr
import json
import time
import threading
from typing import Dict
import uuid

# Store active jobs and their status
jobs: Dict[str, dict] = {}

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
    demo.launch() 