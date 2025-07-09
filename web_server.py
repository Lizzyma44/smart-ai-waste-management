#!/usr/bin/env python3
"""
Web server for live waste detection
"""
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import threading
import time
from datetime import datetime

from waste_classifier import WasteClassifier
from data_manager import DataManager

app = Flask(__name__)

# Global variables
classifier = WasteClassifier()
data_manager = DataManager()
classification_count = 0

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify image from webcam"""
    global classification_count
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV image
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Classify image
        waste_type, confidence, processing_time = classifier.classify_waste(image_cv)
        
        if waste_type:
            # Update count
            classification_count += 1
            
            # Save to database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = None
            
            if data_manager.get_config('classification').get('save_images', True):
                image_path = data_manager.save_image(image_cv, timestamp, waste_type)
            
            data_manager.save_classification(waste_type, confidence, image_path, processing_time)
            
            return jsonify({
                'success': True,
                'waste_type': waste_type,
                'confidence': confidence,
                'processing_time': processing_time,
                'count': classification_count,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Classification failed'
            })
            
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stats')
def get_stats():
    """Get classification statistics"""
    try:
        stats = data_manager.get_classification_stats()
        recent = data_manager.get_recent_classifications(10)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent': recent,
            'total_count': classification_count
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Starting Waste Detection Web Server...")
    print("Open your browser and go to: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    # Create templates directory if it doesn't exist
    import os
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=8080, debug=True)