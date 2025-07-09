#!/usr/bin/env python3
"""
Simple web server for waste detection
"""
import http.server
import socketserver
import webbrowser
import threading
import time
import json
import urllib.parse
import base64
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

from waste_classifier import WasteClassifier
from data_manager import DataManager

# Initialize components
classifier = WasteClassifier()
data_manager = DataManager()
classification_count = 0

class WasteDetectionHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Read the HTML file
            try:
                with open('templates/index.html', 'r') as f:
                    html_content = f.read()
                self.wfile.write(html_content.encode())
            except FileNotFoundError:
                self.wfile.write(b'<h1>Error: Template not found</h1>')
        
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                stats = data_manager.get_classification_stats()
                recent = data_manager.get_recent_classifications(10)
                
                response = {
                    'success': True,
                    'stats': stats,
                    'recent': recent,
                    'total_count': classification_count
                }
                
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        
        else:
            super().do_GET()
    
    def do_POST(self):
        global classification_count
        
        if self.path == '/classify':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                image_data = data['image']
                
                # Decode base64 image
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Convert to OpenCV image
                image = Image.open(io.BytesIO(image_bytes))
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Classify image
                waste_type, confidence, processing_time = classifier.classify_waste(image_cv)
                
                if waste_type:
                    classification_count += 1
                    
                    # Save to database
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = None
                    
                    if data_manager.get_config('classification').get('save_images', True):
                        image_path = data_manager.save_image(image_cv, timestamp, waste_type)
                    
                    data_manager.save_classification(waste_type, confidence, image_path, processing_time)
                    
                    response = {
                        'success': True,
                        'waste_type': waste_type,
                        'confidence': confidence,
                        'processing_time': processing_time,
                        'count': classification_count,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    response = {'success': False, 'error': 'Classification failed'}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_server():
    PORT = 8080
    
    # Try different ports if 8080 is in use
    for port in range(8080, 8090):
        try:
            with socketserver.TCPServer(("", port), WasteDetectionHandler) as httpd:
                print(f"Starting server on port {port}")
                print(f"Open your browser and go to: http://localhost:{port}")
                print("Press Ctrl+C to stop the server")
                
                # Auto-open browser
                threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
                
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"Port {port} is in use, trying next port...")
                continue
            else:
                raise
    
    print("Could not find an available port")

if __name__ == '__main__':
    start_server()