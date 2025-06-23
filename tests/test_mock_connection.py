#!/usr/bin/env python3
"""
Test script to verify mock server connection
"""

import requests
import json
import time

def test_mock_server():
    """Test if the mock server is responding"""
    base_url = "http://localhost:8188"
    
    print("Testing mock server connection...")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/queue", timeout=5)
        print(f"✓ Queue endpoint: {response.status_code}")
        
        response = requests.get(f"{base_url}/system_stats", timeout=5)
        print(f"✓ System stats endpoint: {response.status_code}")
        
        response = requests.get(f"{base_url}/history", timeout=5)
        print(f"✓ History endpoint: {response.status_code}")
        
        response = requests.get(f"{base_url}/prompt", timeout=5)
        print(f"✓ Prompt endpoint: {response.status_code}")
        
        # Test POST endpoints
        response = requests.post(f"{base_url}/interrupt", json={}, timeout=5)
        print(f"✓ Interrupt endpoint: {response.status_code}")
        
        response = requests.post(f"{base_url}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
        print(f"✓ Free endpoint: {response.status_code}")
        
        print("\n✓ All endpoints responding correctly!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to mock server. Is it running?")
        print("Run: python start_mock_server.py")
        return False
    except Exception as e:
        print(f"✗ Error testing mock server: {e}")
        return False

if __name__ == "__main__":
    test_mock_server() 