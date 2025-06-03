import asyncio
from gradio_client import Client
import time

class TestClient:
    def __init__(self, host: str = "http://localhost:7860"):
        self.client = Client(host)
        
    async def _poll_until_complete(self, job_id: str) -> str:
        """Poll for job completion using HTTP"""
        print(f"Starting HTTP polling for job {job_id}")
        attempts = 0
        while True:
            try:
                result = self.client.predict(job_id, api_name="/workflow_result")
                print(f"Polling job {job_id}: {result}")
                
                if result["status"] == "completed":
                    if result.get("output_file"):
                        return result["output_file"]
                    raise Exception(f"Job {job_id} completed but no output file found")
                elif result["status"] == "failed":
                    raise Exception(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"Error polling job {job_id}: {str(e)}")
                
            attempts += 1
            sleep_time = min(2.0 * (1.2 ** (attempts - 1)), 5.0)
            await asyncio.sleep(sleep_time)
            
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