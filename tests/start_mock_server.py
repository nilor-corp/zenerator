#!/usr/bin/env python3
"""
Simple script to start the mock ComfyUI server
"""

import sys
import os

# Add the tests directory to the path so we can import the mock server
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

from mock_comfy_ws_server import create_mock_server
import time

def main():
    print("Starting mock ComfyUI server...")
    server = create_mock_server()
    
    try:
        print("Mock server is running on http://localhost:8188")
        print("WebSocket available at ws://localhost:8188/ws")
        print("Press Ctrl+C to stop the server")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down mock server...")
        server.stop()
        print("Mock server stopped")

if __name__ == "__main__":
    main() 