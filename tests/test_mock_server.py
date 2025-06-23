#!/usr/bin/env python3
"""
Test script for the mock ComfyUI server
"""

import requests
import json
import time
import threading
from mock_comfy_ws_server import create_mock_server

def test_http_endpoints():
    """Test all HTTP endpoints"""
    base_url = "http://localhost:8188"
    
    # Test GET endpoints
    print("Testing GET endpoints...")
    
    # Test /queue
    response = requests.get(f"{base_url}/queue")
    print(f"/queue: {response.status_code} - {response.json()}")
    
    # Test /history
    response = requests.get(f"{base_url}/history")
    print(f"/history: {response.status_code} - {response.json()}")
    
    # Test /system_stats
    response = requests.get(f"{base_url}/system_stats")
    print(f"/system_stats: {response.status_code} - {response.json()}")
    
    # Test /prompt
    response = requests.get(f"{base_url}/prompt")
    print(f"/prompt: {response.status_code} - {response.json()}")
    
    # Test POST endpoints
    print("\nTesting POST endpoints...")
    
    # Test /prompt
    workflow_data = {"test": "workflow"}
    response = requests.post(f"{base_url}/prompt", json=workflow_data)
    print(f"/prompt POST: {response.status_code} - {response.json()}")
    
    # Test /interrupt
    response = requests.post(f"{base_url}/interrupt", json={})
    print(f"/interrupt: {response.status_code} - {response.json()}")
    
    # Test /history clear
    response = requests.post(f"{base_url}/history", json={"clear": True})
    print(f"/history clear: {response.status_code} - {response.json()}")
    
    # Test /free
    response = requests.post(f"{base_url}/free", json={"unload_models": True, "free_memory": True})
    print(f"/free: {response.status_code} - {response.json()}")

def main():
    print("Starting mock server...")
    server = create_mock_server()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        test_http_endpoints()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        print("Stopping mock server...")
        server.stop()

if __name__ == "__main__":
    main() 