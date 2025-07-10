# 🚀 Raspberry Pi 5 Deployment Guide
## Smart AI Waste Management System

*Complete step-by-step guide for deploying the web-based AI waste management system to Raspberry Pi 5*

---

## 📋 Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [OS Installation & Setup](#os-installation--setup)
3. [Environment Setup](#environment-setup)
4. [Camera Integration](#camera-integration)
5. [AI Model Optimization](#ai-model-optimization)
6. [Web Interface Adaptation](#web-interface-adaptation)
7. [Hardware Integration](#hardware-integration)
8. [Performance Optimization](#performance-optimization)
9. [Testing & Validation](#testing--validation)
10. [Deployment Package](#deployment-package)
11. [Technical Implementation](#technical-implementation)
12. [Troubleshooting Guide](#troubleshooting-guide)
13. [Maintenance Procedures](#maintenance-procedures)

---

## 🛠️ Hardware Requirements

### Essential Components

| Component | Specification | Purpose | Est. Cost |
|-----------|--------------|---------|-----------|
| **Raspberry Pi 5** | 8GB RAM model | Main processing unit | $80 |
| **Camera Module 3** | 12MP, autofocus | High-quality image capture | $25 |
| **7" Touchscreen** | Official RPi display | User interface | $60 |
| **microSD Card** | 64GB Class 10 U3 | Storage (min 32GB) | $15 |
| **Power Supply** | Official 27W USB-C | Stable power delivery | $12 |
| **Cooling Solution** | Active fan or heat sink | Thermal management | $10 |

### Optional Components

| Component | Purpose | Est. Cost |
|-----------|---------|-----------|
| **Case/Enclosure** | Protection and mounting | $20-50 |
| **GPIO Breakout** | Hardware integration | $10 |
| **LED Strip** | Consistent lighting | $15 |
| **Arduino Uno** | External hardware control | $25 |
| **Ultrasonic Sensor** | Presence detection | $5 |

### **Total Hardware Cost: ~$200-250**

---

## 💽 OS Installation & Setup

### Step 1: Flash Raspberry Pi OS

```bash
# Download Raspberry Pi Imager
# https://www.raspberrypi.org/software/

# Flash Raspberry Pi OS (64-bit) to microSD card
# Enable SSH, VNC, and Camera in advanced options
# Set username: pi, password: your_password
```

### Step 2: Initial Boot Configuration

```bash
# SSH into your Pi
ssh pi@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera and other interfaces
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → SSH → Enable
# Interface Options → VNC → Enable
# Advanced Options → Expand Filesystem

# Reboot
sudo reboot
```

### Step 3: System Configuration

```bash
# Install essential packages
sudo apt install -y git curl wget vim htop tree
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test

# Set GPU memory split (important for camera and AI)
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt

# Optimize for performance
echo 'arm_boost=1' | sudo tee -a /boot/config.txt
echo 'over_voltage=2' | sudo tee -a /boot/config.txt
echo 'arm_freq=2400' | sudo tee -a /boot/config.txt

sudo reboot
```

---

## 🐍 Environment Setup

### Step 1: Create Project Environment

```bash
# Clone the repository
git clone https://github.com/raviramp36/smart-ai-waste-management.git
cd smart-ai-waste-management

# Create virtual environment
python3 -m venv venv_rpi
source venv_rpi/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install RPi-Optimized Dependencies

```bash
# Create RPi-specific requirements file
cat > requirements_rpi.txt << EOF
# Core dependencies optimized for RPi
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.0.0

# TensorFlow Lite (lightweight for RPi)
tflite-runtime==2.13.0

# Camera support for RPi
picamera2
libcamera

# Web framework
flask==2.3.3
flask-cors==4.0.0
websockets==11.0.3

# Hardware interfaces
RPi.GPIO==0.7.1
pyserial==3.5

# Utilities
psutil==5.9.5
python-dotenv==1.0.0
schedule==1.2.0
EOF

# Install dependencies
pip install -r requirements_rpi.txt
```

### Step 3: System Optimization

```bash
# Increase swap file for compilation
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Optimize Python for ARM
export OPENBLAS_CORETYPE=ARMV8
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
```

---

## 📷 Camera Integration

### Step 1: Camera Hardware Setup

```bash
# Connect Camera Module 3 to RPi 5 camera port
# Verify camera detection
libcamera-hello --list-cameras

# Test camera capture
libcamera-jpeg -o test.jpg --width 1920 --height 1080
```

### Step 2: Create RPi Camera Manager

```python
# Create src/rpi_camera_manager.py
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import threading

class RPiCameraManager:
    """
    Raspberry Pi Camera Manager using PiCamera2
    Optimized for waste detection with auto-focus and exposure control
    """
    
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.picam2 = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize camera with optimal settings for waste detection"""
        try:
            self.picam2 = Picamera2()
            
            # Configure camera for high-quality stills
            still_config = self.picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                lores={"size": (640, 480), "format": "YUV420"}
            )
            
            self.picam2.configure(still_config)
            
            # Set optimal controls for waste detection
            controls_dict = {
                "AfMode": controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
                "AeEnable": True,
                "AwbEnable": True,
                "Brightness": 0.0,
                "Contrast": 1.2,
                "Saturation": 1.1
            }
            
            self.picam2.set_controls(controls_dict)
            self.picam2.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            print(f"RPi Camera initialized: {self.resolution[0]}x{self.resolution[1]}")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def start_capture(self):
        """Start continuous frame capture"""
        if not self.picam2:
            if not self.initialize():
                return False
                
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Continuous capture loop"""
        while self.running:
            try:
                # Capture frame from camera
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.frame = frame_bgr.copy()
                    
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
            
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def capture_image(self):
        """Capture single high-quality image for classification"""
        try:
            # Use high-resolution capture for classification
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return frame_bgr, timestamp
            
        except Exception as e:
            print(f"Image capture error: {e}")
            return None, None
    
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
    
    def __del__(self):
        """Cleanup"""
        self.stop_capture()
```

### Step 3: Test Camera Integration

```python
# Create test_rpi_camera.py
from src.rpi_camera_manager import RPiCameraManager
import cv2

def test_camera():
    camera = RPiCameraManager()
    
    if camera.initialize():
        print("Camera initialized successfully")
        
        # Test single capture
        frame, timestamp = camera.capture_image()
        if frame is not None:
            cv2.imwrite(f"test_capture_{timestamp}.jpg", frame)
            print(f"Test image saved: test_capture_{timestamp}.jpg")
        
        # Test continuous capture
        camera.start_capture()
        
        for i in range(10):
            frame = camera.get_frame()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}")
            time.sleep(1)
        
        camera.stop_capture()
        print("Camera test completed")
    else:
        print("Camera initialization failed")

if __name__ == "__main__":
    test_camera()
```

---

## 🧠 AI Model Optimization

### Step 1: Model Conversion Tools

```bash
# Install TensorFlow for model conversion
pip install tensorflow==2.13.0

# Create model conversion script
mkdir models_rpi
```

### Step 2: Convert Models to TensorFlow Lite

```python
# Create scripts/convert_models.py
import tensorflow as tf
import urllib.request
import os

def convert_mobilenet_to_tflite():
    """Convert MobileNet model to TensorFlow Lite"""
    print("Converting MobileNet to TensorFlow Lite...")
    
    # Load pre-trained MobileNetV2
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        classes=1000
    )
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Quantization for better performance
    converter.representative_dataset = generate_representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save the model
    with open('models_rpi/mobilenet_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("MobileNet conversion completed: models_rpi/mobilenet_quantized.tflite")

def generate_representative_dataset():
    """Generate representative dataset for quantization"""
    import numpy as np
    for _ in range(100):
        # Generate random data that represents your input
        yield [np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)]

def download_and_convert_coco_ssd():
    """Download and convert COCO-SSD model"""
    print("Downloading COCO-SSD model...")
    
    # Download pre-converted COCO-SSD Lite model
    url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    
    import zipfile
    urllib.request.urlretrieve(url, "coco_ssd.zip")
    
    with zipfile.ZipFile("coco_ssd.zip", 'r') as zip_ref:
        zip_ref.extractall("models_rpi/")
    
    os.remove("coco_ssd.zip")
    print("COCO-SSD model ready: models_rpi/detect.tflite")

if __name__ == "__main__":
    convert_mobilenet_to_tflite()
    download_and_convert_coco_ssd()
```

### Step 3: Create TensorFlow Lite Classifier

```python
# Create src/tflite_classifier.py
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

class TFLiteWasteClassifier:
    """
    TensorFlow Lite optimized waste classifier for Raspberry Pi
    """
    
    def __init__(self, mobilenet_path, coco_path):
        self.mobilenet_interpreter = None
        self.coco_interpreter = None
        self.mobilenet_path = mobilenet_path
        self.coco_path = coco_path
        
        # Waste classification mapping
        self.waste_classification = {
            # Wet/Organic Waste
            'banana': 'wet', 'apple': 'wet', 'orange': 'wet', 'broccoli': 'wet',
            'carrot': 'wet', 'hot dog': 'wet', 'pizza': 'wet', 'donut': 'wet',
            'cake': 'wet', 'sandwich': 'wet', 'food': 'wet', 'fruit': 'wet',
            
            # Dry/Recyclable Waste
            'bottle': 'dry', 'wine glass': 'dry', 'cup': 'dry', 'bowl': 'dry',
            'cell phone': 'dry', 'laptop': 'dry', 'book': 'dry', 'scissors': 'dry',
            'plastic': 'dry', 'glass': 'dry', 'metal': 'dry', 'paper': 'dry'
        }
        
        # COCO class names
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def load_models(self):
        """Load TensorFlow Lite models"""
        try:
            # Load MobileNet interpreter
            self.mobilenet_interpreter = tflite.Interpreter(
                model_path=self.mobilenet_path,
                num_threads=4  # Use multiple cores
            )
            self.mobilenet_interpreter.allocate_tensors()
            
            # Load COCO-SSD interpreter
            self.coco_interpreter = tflite.Interpreter(
                model_path=self.coco_path,
                num_threads=4
            )
            self.coco_interpreter.allocate_tensors()
            
            print("TensorFlow Lite models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Model loading error: {e}")
            return False
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model input"""
        # Resize image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(pil_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        return img_array
    
    def classify_with_mobilenet(self, image):
        """Classify image using MobileNet"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(image)
            
            # Get input and output details
            input_details = self.mobilenet_interpreter.get_input_details()
            output_details = self.mobilenet_interpreter.get_output_details()
            
            # Set input tensor
            self.mobilenet_interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self.mobilenet_interpreter.invoke()
            
            # Get output
            output_data = self.mobilenet_interpreter.get_tensor(output_details[0]['index'])
            predictions = output_data[0]
            
            # Get top prediction
            top_index = np.argmax(predictions)
            confidence = predictions[top_index]
            
            # Note: This would require ImageNet class mapping
            # For simplicity, we'll use a generic classification
            return "general_object", confidence
            
        except Exception as e:
            print(f"MobileNet classification error: {e}")
            return None, 0.0
    
    def detect_with_coco(self, image):
        """Detect objects using COCO-SSD"""
        try:
            # Preprocess image for COCO-SSD (typically 300x300)
            input_data = self.preprocess_image(image, target_size=(300, 300))
            input_data = (input_data * 255).astype(np.uint8)
            
            # Get input and output details
            input_details = self.coco_interpreter.get_input_details()
            output_details = self.coco_interpreter.get_output_details()
            
            # Set input tensor
            self.coco_interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self.coco_interpreter.invoke()
            
            # Get outputs
            boxes = self.coco_interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.coco_interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.coco_interpreter.get_tensor(output_details[2]['index'])[0]
            
            # Filter detections by confidence threshold
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.5:  # Confidence threshold
                    class_id = int(classes[i])
                    if class_id < len(self.coco_classes):
                        detections.append({
                            'class': self.coco_classes[class_id],
                            'score': float(scores[i]),
                            'bbox': boxes[i].tolist()
                        })
            
            return detections
            
        except Exception as e:
            print(f"COCO detection error: {e}")
            return []
    
    def classify_waste(self, image):
        """Main classification function"""
        start_time = time.time()
        
        # Get object detections
        detections = self.detect_with_coco(image)
        
        # Classify with MobileNet as fallback
        mobilenet_class, mobilenet_confidence = self.classify_with_mobilenet(image)
        
        # Process results
        detected_object = "unknown"
        confidence = 0.0
        
        if detections:
            # Use highest confidence detection
            best_detection = max(detections, key=lambda x: x['score'])
            detected_object = best_detection['class']
            confidence = best_detection['score']
        elif mobilenet_confidence > 0.3:
            detected_object = mobilenet_class
            confidence = mobilenet_confidence
        
        # Determine waste category
        waste_category = self.determine_waste_type(detected_object)
        
        processing_time = time.time() - start_time
        
        return {
            'detected_object': detected_object,
            'waste_category': waste_category,
            'confidence': confidence,
            'processing_time': processing_time,
            'all_detections': detections
        }
    
    def determine_waste_type(self, detected_object):
        """Determine if object is wet or dry waste"""
        if detected_object in self.waste_classification:
            category = self.waste_classification[detected_object]
            return "WET WASTE" if category == 'wet' else "DRY WASTE"
        
        # Fallback logic for unknown objects
        object_lower = detected_object.lower()
        
        # Check for keyword matches
        wet_keywords = ['food', 'fruit', 'vegetable', 'organic', 'eat', 'banana', 'apple']
        dry_keywords = ['bottle', 'can', 'glass', 'plastic', 'metal', 'paper', 'electronic']
        
        if any(keyword in object_lower for keyword in wet_keywords):
            return "WET WASTE"
        elif any(keyword in object_lower for keyword in dry_keywords):
            return "DRY WASTE"
        else:
            return "DRY WASTE"  # Default to dry waste
```

### Step 4: Run Model Conversion

```bash
# Convert models
cd smart-ai-waste-management
python scripts/convert_models.py

# Verify models
ls -la models_rpi/
# Should show: mobilenet_quantized.tflite, detect.tflite
```

---

## 🌐 Web Interface Adaptation

### Step 1: Create Flask Web Server

```python
# Create src/web_server.py
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import json
import threading
import time
from src.rpi_camera_manager import RPiCameraManager
from src.tflite_classifier import TFLiteWasteClassifier
import serial
import serial.tools.list_ports

app = Flask(__name__)
CORS(app)

# Global variables
camera_manager = None
classifier = None
arduino_serial = None
current_result = {"waste_category": "READY", "confidence": 0, "detected_object": "none"}

def initialize_system():
    """Initialize camera and AI models"""
    global camera_manager, classifier
    
    print("Initializing RPi system...")
    
    # Initialize camera
    camera_manager = RPiCameraManager()
    if not camera_manager.start_capture():
        print("Failed to initialize camera")
        return False
    
    # Initialize AI classifier
    classifier = TFLiteWasteClassifier(
        mobilenet_path="models_rpi/mobilenet_quantized.tflite",
        coco_path="models_rpi/detect.tflite"
    )
    
    if not classifier.load_models():
        print("Failed to load AI models")
        return False
    
    print("System initialized successfully")
    return True

def generate_frames():
    """Generate video frames for streaming"""
    global camera_manager
    
    while True:
        if camera_manager:
            frame = camera_manager.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Main interface"""
    return render_template('rpi_interface.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_waste():
    """Classify current camera frame"""
    global camera_manager, classifier, current_result, arduino_serial
    
    if not camera_manager or not classifier:
        return jsonify({"error": "System not initialized"})
    
    # Capture high-resolution image
    image, timestamp = camera_manager.capture_image()
    if image is None:
        return jsonify({"error": "Failed to capture image"})
    
    # Classify waste
    result = classifier.classify_waste(image)
    current_result = result
    
    # Send command to Arduino if connected
    if arduino_serial and arduino_serial.is_open:
        try:
            command = 'w' if result['waste_category'] == 'WET WASTE' else 'd'
            arduino_serial.write(command.encode())
            result['arduino_command'] = command
        except Exception as e:
            print(f"Arduino communication error: {e}")
    
    return jsonify(result)

@app.route('/arduino/connect', methods=['POST'])
def connect_arduino():
    """Connect to Arduino via serial"""
    global arduino_serial
    
    data = request.get_json()
    port = data.get('port', '/dev/ttyUSB0')
    baud_rate = data.get('baud_rate', 9600)
    
    try:
        arduino_serial = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Allow Arduino to reset
        return jsonify({"status": "connected", "port": port})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/arduino/disconnect', methods=['POST'])
def disconnect_arduino():
    """Disconnect Arduino"""
    global arduino_serial
    
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
        arduino_serial = None
    
    return jsonify({"status": "disconnected"})

@app.route('/arduino/ports')
def list_arduino_ports():
    """List available serial ports"""
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports})

@app.route('/status')
def system_status():
    """Get system status"""
    status = {
        "camera": camera_manager is not None,
        "classifier": classifier is not None,
        "arduino": arduino_serial is not None and arduino_serial.is_open if arduino_serial else False,
        "current_result": current_result
    }
    return jsonify(status)

if __name__ == '__main__':
    if initialize_system():
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("Failed to initialize system")
```

### Step 2: Create RPi-Optimized HTML Template

```html
<!-- Create templates/rpi_interface.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Waste Detection - Raspberry Pi</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
            touch-action: manipulation; /* Optimize for touch */
        }

        .main-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            height: 100vh;
            gap: 20px;
            padding: 20px;
        }

        .camera-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            overflow: hidden;
            position: relative;
        }

        #videoFeed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .controls-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .status-panel, .results-panel, .arduino-panel {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            color: white;
        }

        .waste-result {
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            font-size: 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .waste-result.wet {
            background: linear-gradient(135deg, #ff9800, #f57c00);
        }

        .waste-result.dry {
            background: linear-gradient(135deg, #4caf50, #2e7d32);
        }

        .waste-result.ready {
            background: rgba(255, 255, 255, 0.2);
        }

        .button {
            background: linear-gradient(135deg, #4caf50, #2e7d32);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 15px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px 0;
            min-height: 50px; /* Touch-friendly */
        }

        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .button:disabled {
            background: rgba(255,255,255,0.1);
            cursor: not-allowed;
            transform: none;
        }

        .arduino-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #f44336;
        }

        .status-dot.connected {
            background: #4caf50;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }

        .info-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }

        .info-value {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .info-label {
            font-size: 12px;
            opacity: 0.8;
        }

        @media (max-width: 768px) {
            .main-container {
                grid-template-columns: 1fr;
                grid-template-rows: 2fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Camera Feed Section -->
        <div class="camera-section">
            <img id="videoFeed" src="/video_feed" alt="Camera Feed">
            
            <!-- Overlay Controls -->
            <div style="position: absolute; top: 20px; left: 20px; right: 20px; display: flex; justify-content: space-between; align-items: center;">
                <h1 style="color: white; font-size: 24px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                    🤖 Smart Waste Detection
                </h1>
                <div id="systemStatus" style="background: rgba(0,0,0,0.5); padding: 10px 15px; border-radius: 10px; color: white;">
                    Initializing...
                </div>
            </div>
        </div>

        <!-- Controls Section -->
        <div class="controls-section">
            <!-- Results Panel -->
            <div class="results-panel">
                <h3>Detection Result</h3>
                <div id="wasteResult" class="waste-result ready">READY</div>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-value" id="confidenceValue">--</div>
                        <div class="info-label">Confidence</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value" id="processingTime">--</div>
                        <div class="info-label">Time (s)</div>
                    </div>
                </div>
                
                <button id="detectBtn" class="button" onclick="detectWaste()">
                    🔍 DETECT WASTE
                </button>
            </div>

            <!-- Arduino Panel -->
            <div class="arduino-panel">
                <h3>Arduino Control</h3>
                
                <div class="arduino-status">
                    <div id="arduinoStatusDot" class="status-dot"></div>
                    <span id="arduinoStatusText">Disconnected</span>
                </div>
                
                <select id="portSelect" style="width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: none;">
                    <option value="">Select Port</option>
                </select>
                
                <button id="connectBtn" class="button" onclick="toggleArduino()">
                    🔌 CONNECT
                </button>
                
                <div id="lastCommand" style="margin-top: 10px; font-size: 14px; opacity: 0.8;">
                    Last command: None
                </div>
            </div>

            <!-- Status Panel -->
            <div class="status-panel">
                <h3>System Status</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-value" id="cameraStatus">--</div>
                        <div class="info-label">Camera</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value" id="aiStatus">--</div>
                        <div class="info-label">AI Models</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isDetecting = false;
        let arduinoConnected = false;

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            updateSystemStatus();
            loadArduinoPorts();
            
            // Auto-refresh system status
            setInterval(updateSystemStatus, 5000);
        });

        async function updateSystemStatus() {
            try {
                const response = await fetch('/status');
                const status = await response.json();
                
                document.getElementById('cameraStatus').textContent = status.camera ? '✅' : '❌';
                document.getElementById('aiStatus').textContent = status.classifier ? '✅' : '❌';
                document.getElementById('systemStatus').textContent = 
                    status.camera && status.classifier ? '🟢 System Ready' : '🔴 System Error';
                
                arduinoConnected = status.arduino;
                updateArduinoStatus();
                
            } catch (error) {
                console.error('Status update error:', error);
                document.getElementById('systemStatus').textContent = '🔴 Connection Error';
            }
        }

        async function detectWaste() {
            if (isDetecting) return;
            
            isDetecting = true;
            document.getElementById('detectBtn').disabled = true;
            document.getElementById('wasteResult').textContent = 'ANALYZING...';
            document.getElementById('wasteResult').className = 'waste-result ready';
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.error) {
                    throw new Error(result.error);
                }
                
                // Update results
                document.getElementById('wasteResult').textContent = result.waste_category;
                document.getElementById('wasteResult').className = 
                    `waste-result ${result.waste_category === 'WET WASTE' ? 'wet' : 'dry'}`;
                
                document.getElementById('confidenceValue').textContent = 
                    (result.confidence * 100).toFixed(0) + '%';
                document.getElementById('processingTime').textContent = 
                    result.processing_time.toFixed(2);
                
                if (result.arduino_command) {
                    document.getElementById('lastCommand').textContent = 
                        `Last command: ${result.arduino_command} (${result.waste_category})`;
                }
                
            } catch (error) {
                console.error('Detection error:', error);
                document.getElementById('wasteResult').textContent = 'ERROR';
                document.getElementById('wasteResult').className = 'waste-result ready';
            } finally {
                isDetecting = false;
                document.getElementById('detectBtn').disabled = false;
            }
        }

        async function loadArduinoPorts() {
            try {
                const response = await fetch('/arduino/ports');
                const data = await response.json();
                
                const select = document.getElementById('portSelect');
                select.innerHTML = '<option value="">Select Port</option>';
                
                data.ports.forEach(port => {
                    const option = document.createElement('option');
                    option.value = port;
                    option.textContent = port;
                    select.appendChild(option);
                });
                
            } catch (error) {
                console.error('Port loading error:', error);
            }
        }

        async function toggleArduino() {
            const connectBtn = document.getElementById('connectBtn');
            
            if (arduinoConnected) {
                // Disconnect
                try {
                    await fetch('/arduino/disconnect', { method: 'POST' });
                    arduinoConnected = false;
                    updateArduinoStatus();
                } catch (error) {
                    console.error('Disconnect error:', error);
                }
            } else {
                // Connect
                const port = document.getElementById('portSelect').value;
                if (!port) {
                    alert('Please select a port');
                    return;
                }
                
                connectBtn.disabled = true;
                connectBtn.textContent = 'CONNECTING...';
                
                try {
                    const response = await fetch('/arduino/connect', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({port: port})
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        throw new Error(result.error);
                    }
                    
                    arduinoConnected = true;
                    updateArduinoStatus();
                    
                } catch (error) {
                    alert('Connection failed: ' + error.message);
                } finally {
                    connectBtn.disabled = false;
                }
            }
        }

        function updateArduinoStatus() {
            const statusDot = document.getElementById('arduinoStatusDot');
            const statusText = document.getElementById('arduinoStatusText');
            const connectBtn = document.getElementById('connectBtn');
            
            if (arduinoConnected) {
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected';
                connectBtn.textContent = '🔌 DISCONNECT';
            } else {
                statusDot.classList.remove('connected');
                statusText.textContent = 'Disconnected';
                connectBtn.textContent = '🔌 CONNECT';
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.code === 'Space' && !e.target.matches('input, select, button')) {
                e.preventDefault();
                detectWaste();
            }
        });
    </script>
</body>
</html>
```

### Step 3: Setup Nginx for Production

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/waste-management << EOF
server {
    listen 80;
    server_name _;
    
    # Increase client max body size for image uploads
    client_max_body_size 10M;
    
    # Proxy to Flask app
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files
    location /static {
        alias /home/pi/smart-ai-waste-management/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/waste-management /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## 🔌 Hardware Integration

### Step 1: GPIO Setup and Arduino Communication

```python
# Create src/hardware_controller.py
import RPi.GPIO as GPIO
import serial
import time
import threading
from enum import Enum

class LEDColor(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class HardwareController:
    """
    Raspberry Pi hardware controller for waste management system
    Controls LEDs, servos, sensors, and Arduino communication
    """
    
    def __init__(self):
        # GPIO pin assignments
        self.LED_RED = 18      # Red LED for wet waste
        self.LED_GREEN = 19    # Green LED for dry waste
        self.LED_BLUE = 20     # Blue LED for system status
        self.SERVO_PIN = 21    # Servo for sorting mechanism
        self.BUZZER_PIN = 22   # Buzzer for audio feedback
        self.TRIGGER_PIN = 23  # Ultrasonic sensor trigger
        self.ECHO_PIN = 24     # Ultrasonic sensor echo
        
        # Arduino serial connection
        self.arduino_serial = None
        self.arduino_port = "/dev/ttyUSB0"
        self.arduino_baud = 9600
        
        # Initialize GPIO
        self.setup_gpio()
        
    def setup_gpio(self):
        """Initialize GPIO pins"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # LED outputs
        GPIO.setup(self.LED_RED, GPIO.OUT)
        GPIO.setup(self.LED_GREEN, GPIO.OUT)
        GPIO.setup(self.LED_BLUE, GPIO.OUT)
        
        # Servo output
        GPIO.setup(self.SERVO_PIN, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(self.SERVO_PIN, 50)  # 50Hz for servo
        self.servo_pwm.start(7.5)  # Neutral position
        
        # Buzzer output
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        
        # Ultrasonic sensor
        GPIO.setup(self.TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)
        
        # Initialize all LEDs off
        self.turn_off_all_leds()
        
        print("GPIO initialized successfully")
    
    def connect_arduino(self, port=None, baud_rate=None):
        """Connect to Arduino via serial"""
        if port:
            self.arduino_port = port
        if baud_rate:
            self.arduino_baud = baud_rate
            
        try:
            if self.arduino_serial and self.arduino_serial.is_open:
                self.arduino_serial.close()
                
            self.arduino_serial = serial.Serial(
                self.arduino_port, 
                self.arduino_baud, 
                timeout=1
            )
            time.sleep(2)  # Allow Arduino to reset
            
            print(f"Arduino connected on {self.arduino_port}")
            return True
            
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            return False
    
    def disconnect_arduino(self):
        """Disconnect Arduino"""
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
            print("Arduino disconnected")
    
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if self.arduino_serial and self.arduino_serial.is_open:
            try:
                self.arduino_serial.write(command.encode())
                return True
            except Exception as e:
                print(f"Arduino communication error: {e}")
                return False
        return False
    
    def control_led(self, color, state):
        """Control individual LED"""
        pin_map = {
            LEDColor.RED: self.LED_RED,
            LEDColor.GREEN: self.LED_GREEN,
            LEDColor.BLUE: self.LED_BLUE
        }
        
        if color in pin_map:
            GPIO.output(pin_map[color], GPIO.HIGH if state else GPIO.LOW)
    
    def turn_off_all_leds(self):
        """Turn off all LEDs"""
        GPIO.output(self.LED_RED, GPIO.LOW)
        GPIO.output(self.LED_GREEN, GPIO.LOW)
        GPIO.output(self.LED_BLUE, GPIO.LOW)
    
    def indicate_waste_type(self, waste_type):
        """Visual indication for waste type"""
        self.turn_off_all_leds()
        
        if waste_type == "WET WASTE":
            # Red LED for wet waste
            self.control_led(LEDColor.RED, True)
            # Send command to Arduino
            self.send_arduino_command('w')
            # Buzz pattern for wet waste
            self.buzz_pattern([0.5, 0.2, 0.5])
            
        elif waste_type == "DRY WASTE":
            # Green LED for dry waste
            self.control_led(LEDColor.GREEN, True)
            # Send command to Arduino
            self.send_arduino_command('d')
            # Buzz pattern for dry waste
            self.buzz_pattern([0.2, 0.1, 0.2, 0.1, 0.2])
        
        # Auto turn off after 3 seconds
        threading.Timer(3.0, self.turn_off_all_leds).start()
    
    def buzz_pattern(self, pattern):
        """Create buzzer pattern (list of durations)"""
        def buzz_sequence():
            for i, duration in enumerate(pattern):
                if i % 2 == 0:  # Buzz on even indices
                    GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
                else:  # Silent on odd indices
                    GPIO.output(self.BUZZER_PIN, GPIO.LOW)
                time.sleep(duration)
            GPIO.output(self.BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off
        
        threading.Thread(target=buzz_sequence, daemon=True).start()
    
    def move_servo(self, angle):
        """Move servo to specified angle (0-180 degrees)"""
        # Convert angle to duty cycle (2.5% to 12.5% for 0-180 degrees)
        duty_cycle = 2.5 + (angle / 180.0) * 10.0
        self.servo_pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow time for movement
        self.servo_pwm.ChangeDutyCycle(0)  # Stop PWM signal
    
    def sort_waste(self, waste_type):
        """Physical sorting mechanism"""
        if waste_type == "WET WASTE":
            self.move_servo(45)   # Move to wet waste bin
        elif waste_type == "DRY WASTE":
            self.move_servo(135)  # Move to dry waste bin
        
        # Return to neutral position after 2 seconds
        threading.Timer(2.0, lambda: self.move_servo(90)).start()
    
    def measure_distance(self):
        """Measure distance using ultrasonic sensor"""
        try:
            # Send trigger pulse
            GPIO.output(self.TRIGGER_PIN, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.TRIGGER_PIN, GPIO.LOW)
            
            # Measure echo time
            start_time = time.time()
            while GPIO.input(self.ECHO_PIN) == 0:
                start_time = time.time()
                
            while GPIO.input(self.ECHO_PIN) == 1:
                end_time = time.time()
            
            # Calculate distance (speed of sound = 34300 cm/s)
            duration = end_time - start_time
            distance = (duration * 34300) / 2
            
            return distance
            
        except Exception as e:
            print(f"Distance measurement error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        self.disconnect_arduino()
        self.servo_pwm.stop()
        GPIO.cleanup()
        print("Hardware cleanup completed")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
```

### Step 2: Arduino Integration Code

```cpp
// Arduino sketch for waste management system
// File: arduino_waste_controller.ino

// Pin definitions
const int LED_WET = 2;        // LED for wet waste indication
const int LED_DRY = 3;        // LED for dry waste indication
const int SERVO_PIN = 9;      // Servo motor for sorting
const int BUZZER_PIN = 10;    // Buzzer for audio feedback
const int STATUS_LED = 13;    // Built-in LED for status

// Include servo library
#include <Servo.h>
Servo sortingServo;

// Variables
bool systemActive = true;
unsigned long lastCommandTime = 0;
const unsigned long TIMEOUT_MS = 5000; // 5 second timeout

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize pins
  pinMode(LED_WET, OUTPUT);
  pinMode(LED_DRY, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);
  
  // Initialize servo
  sortingServo.attach(SERVO_PIN);
  sortingServo.write(90); // Neutral position
  
  // Startup sequence
  startupSequence();
  
  Serial.println("Arduino Waste Management Controller Ready");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    char command = Serial.read();
    processCommand(command);
    lastCommandTime = millis();
  }
  
  // Status LED heartbeat
  if (millis() % 1000 < 500) {
    digitalWrite(STATUS_LED, HIGH);
  } else {
    digitalWrite(STATUS_LED, LOW);
  }
  
  // Timeout check
  if (millis() - lastCommandTime > TIMEOUT_MS && lastCommandTime > 0) {
    // Reset system if no commands received
    resetSystem();
    lastCommandTime = 0;
  }
  
  delay(50); // Small delay for stability
}

void processCommand(char command) {
  switch (command) {
    case 'w':
    case 'W':
      handleWetWaste();
      break;
      
    case 'd':
    case 'D':
      handleDryWaste();
      break;
      
    case 'r':
    case 'R':
      resetSystem();
      break;
      
    case 's':
    case 'S':
      sendStatus();
      break;
      
    default:
      Serial.println("Unknown command");
      break;
  }
}

void handleWetWaste() {
  Serial.println("Processing wet waste...");
  
  // Turn on wet waste LED
  digitalWrite(LED_WET, HIGH);
  digitalWrite(LED_DRY, LOW);
  
  // Move servo to wet waste position (45 degrees)
  sortingServo.write(45);
  
  // Buzzer pattern for wet waste (long beep)
  tone(BUZZER_PIN, 1000, 500);
  delay(600);
  
  // Hold position for 2 seconds
  delay(2000);
  
  // Return to neutral
  returnToNeutral();
  
  Serial.println("Wet waste processed");
}

void handleDryWaste() {
  Serial.println("Processing dry waste...");
  
  // Turn on dry waste LED
  digitalWrite(LED_DRY, HIGH);
  digitalWrite(LED_WET, LOW);
  
  // Move servo to dry waste position (135 degrees)
  sortingServo.write(135);
  
  // Buzzer pattern for dry waste (two short beeps)
  tone(BUZZER_PIN, 1500, 200);
  delay(300);
  tone(BUZZER_PIN, 1500, 200);
  delay(300);
  
  // Hold position for 2 seconds
  delay(2000);
  
  // Return to neutral
  returnToNeutral();
  
  Serial.println("Dry waste processed");
}

void returnToNeutral() {
  // Turn off all LEDs
  digitalWrite(LED_WET, LOW);
  digitalWrite(LED_DRY, LOW);
  
  // Return servo to neutral position
  sortingServo.write(90);
  
  delay(500); // Allow servo to move
}

void resetSystem() {
  Serial.println("System reset");
  
  // Turn off all outputs
  digitalWrite(LED_WET, LOW);
  digitalWrite(LED_DRY, LOW);
  noTone(BUZZER_PIN);
  
  // Return servo to neutral
  sortingServo.write(90);
  
  // Brief indication
  digitalWrite(STATUS_LED, HIGH);
  delay(100);
  digitalWrite(STATUS_LED, LOW);
}

void sendStatus() {
  // Send system status over serial
  Serial.println("STATUS:OK");
  Serial.print("SERVO_POS:");
  Serial.println(sortingServo.read());
  Serial.print("WET_LED:");
  Serial.println(digitalRead(LED_WET));
  Serial.print("DRY_LED:");
  Serial.println(digitalRead(LED_DRY));
}

void startupSequence() {
  // Visual startup sequence
  digitalWrite(LED_WET, HIGH);
  delay(300);
  digitalWrite(LED_DRY, HIGH);
  delay(300);
  digitalWrite(LED_WET, LOW);
  delay(300);
  digitalWrite(LED_DRY, LOW);
  
  // Servo sweep
  for (int pos = 90; pos <= 135; pos++) {
    sortingServo.write(pos);
    delay(10);
  }
  for (int pos = 135; pos >= 45; pos--) {
    sortingServo.write(pos);
    delay(10);
  }
  for (int pos = 45; pos <= 90; pos++) {
    sortingServo.write(pos);
    delay(10);
  }
  
  // Startup sound
  tone(BUZZER_PIN, 2000, 100);
  delay(150);
  tone(BUZZER_PIN, 2500, 100);
  delay(150);
  tone(BUZZER_PIN, 3000, 100);
}
```

### Step 3: Create Hardware Test Script

```python
# Create scripts/test_hardware.py
from src.hardware_controller import HardwareController
import time

def test_hardware_components():
    """Test all hardware components"""
    print("🧪 Testing Raspberry Pi Hardware Components...")
    
    controller = HardwareController()
    
    try:
        # Test LEDs
        print("\n1. Testing LEDs...")
        for color in ['RED', 'GREEN', 'BLUE']:
            print(f"   Testing {color} LED")
            if color == 'RED':
                controller.control_led(controller.LEDColor.RED, True)
            elif color == 'GREEN':
                controller.control_led(controller.LEDColor.GREEN, True)
            elif color == 'BLUE':
                controller.control_led(controller.LEDColor.BLUE, True)
            time.sleep(1)
            controller.turn_off_all_leds()
            time.sleep(0.5)
        
        # Test buzzer patterns
        print("\n2. Testing Buzzer...")
        print("   Wet waste pattern")
        controller.buzz_pattern([0.5])
        time.sleep(2)
        print("   Dry waste pattern")
        controller.buzz_pattern([0.2, 0.1, 0.2])
        time.sleep(2)
        
        # Test servo
        print("\n3. Testing Servo...")
        positions = [45, 90, 135, 90]  # Wet, neutral, dry, neutral
        for pos in positions:
            print(f"   Moving to {pos} degrees")
            controller.move_servo(pos)
            time.sleep(1)
        
        # Test ultrasonic sensor
        print("\n4. Testing Ultrasonic Sensor...")
        for i in range(5):
            distance = controller.measure_distance()
            if distance:
                print(f"   Distance measurement {i+1}: {distance:.2f} cm")
            else:
                print(f"   Distance measurement {i+1}: Failed")
            time.sleep(1)
        
        # Test Arduino connection
        print("\n5. Testing Arduino Connection...")
        if controller.connect_arduino():
            print("   Arduino connected successfully")
            
            # Test commands
            controller.send_arduino_command('s')  # Status
            time.sleep(1)
            controller.send_arduino_command('w')  # Wet waste
            time.sleep(3)
            controller.send_arduino_command('d')  # Dry waste
            time.sleep(3)
            controller.send_arduino_command('r')  # Reset
            
            controller.disconnect_arduino()
            print("   Arduino test completed")
        else:
            print("   Arduino connection failed (check connection)")
        
        # Test complete waste indication
        print("\n6. Testing Complete Waste Indication...")
        controller.indicate_waste_type("WET WASTE")
        time.sleep(4)
        controller.indicate_waste_type("DRY WASTE")
        time.sleep(4)
        
        print("\n✅ Hardware test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Hardware test failed: {e}")
    
    finally:
        controller.cleanup()

if __name__ == "__main__":
    test_hardware_components()
```

---

## 🚀 Performance Optimization

### Step 1: System Configuration

```bash
# Create scripts/optimize_rpi.sh
#!/bin/bash

echo "🚀 Optimizing Raspberry Pi 5 for AI workloads..."

# GPU Memory Split
echo "Setting GPU memory split to 128MB..."
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt

# CPU Governor
echo "Setting CPU governor to performance..."
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable unnecessary services
echo "Disabling unnecessary services..."
sudo systemctl disable bluetooth
sudo systemctl disable hciuart
sudo systemctl disable cups
sudo systemctl disable avahi-daemon

# Enable hardware acceleration
echo "Enabling hardware acceleration..."
echo 'dtoverlay=vc4-kms-v3d' | sudo tee -a /boot/config.txt
echo 'max_framebuffers=2' | sudo tee -a /boot/config.txt

# Increase swap file
echo "Optimizing swap configuration..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Network optimizations
echo "Applying network optimizations..."
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

# I/O Scheduler optimization
echo "Setting I/O scheduler to deadline..."
echo 'deadline' | sudo tee /sys/block/mmcblk0/queue/scheduler

echo "✅ Optimization completed. Reboot required."
```

### Step 2: TensorFlow Lite Optimization

```python
# Create src/model_optimizer.py
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import threading

class OptimizedTFLiteInference:
    """
    Optimized TensorFlow Lite inference for Raspberry Pi 5
    Uses multithreading and efficient memory management
    """
    
    def __init__(self, model_path, num_threads=4):
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.lock = threading.Lock()
        
        self.load_model()
    
    def load_model(self):
        """Load and configure TensorFlow Lite model"""
        try:
            # Configure for optimal performance
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )
            
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            
            print(f"Model loaded: {self.model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Using {self.num_threads} threads")
            
        except Exception as e:
            print(f"Model loading error: {e}")
            raise
    
    def preprocess_image_optimized(self, image):
        """Optimized image preprocessing"""
        target_size = (self.input_shape[2], self.input_shape[1])
        
        # Use OpenCV for faster resizing
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and prepare for inference
        if self.input_details[0]['dtype'] == np.uint8:
            # Integer input
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            # Float input
            input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        
        return input_data
    
    def run_inference(self, input_data):
        """Run optimized inference"""
        with self.lock:  # Thread safety
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return output_data, inference_time
    
    def classify_image(self, image):
        """Complete classification pipeline"""
        # Preprocess
        input_data = self.preprocess_image_optimized(image)
        
        # Inference
        output_data, inference_time = self.run_inference(input_data)
        
        return output_data, inference_time

class PerformanceMonitor:
    """Monitor system performance during inference"""
    
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.cpu_temps = []
    
    def log_inference(self, inference_time):
        """Log inference performance"""
        self.inference_times.append(inference_time)
        
        # Keep only last 100 measurements
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
    
    def get_average_inference_time(self):
        """Get average inference time"""
        if self.inference_times:
            return sum(self.inference_times) / len(self.inference_times)
        return 0
    
    def get_fps(self):
        """Calculate effective FPS"""
        avg_time = self.get_average_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def monitor_system(self):
        """Monitor system resources"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Temperature (RPi specific)
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read()) / 1000.0
            except:
                temp = 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'temperature': temp,
                'fps': self.get_fps()
            }
            
        except ImportError:
            return {'error': 'psutil not installed'}
```

### Step 3: Create Performance Benchmark

```python
# Create scripts/benchmark_performance.py
import time
import cv2
import numpy as np
from src.rpi_camera_manager import RPiCameraManager
from src.tflite_classifier import TFLiteWasteClassifier
from src.model_optimizer import OptimizedTFLiteInference, PerformanceMonitor

def benchmark_system():
    """Comprehensive system performance benchmark"""
    print("🏁 Starting Raspberry Pi 5 Performance Benchmark...")
    
    monitor = PerformanceMonitor()
    
    # Test 1: Model Loading Speed
    print("\n1. Model Loading Performance")
    start_time = time.time()
    classifier = TFLiteWasteClassifier(
        mobilenet_path="models_rpi/mobilenet_quantized.tflite",
        coco_path="models_rpi/detect.tflite"
    )
    
    if classifier.load_models():
        load_time = time.time() - start_time
        print(f"   ✅ Models loaded in {load_time:.2f} seconds")
    else:
        print("   ❌ Model loading failed")
        return
    
    # Test 2: Camera Performance
    print("\n2. Camera Performance")
    camera = RPiCameraManager()
    if camera.initialize():
        camera.start_capture()
        
        # Test frame capture speed
        frame_times = []
        for i in range(30):
            start = time.time()
            frame = camera.get_frame()
            if frame is not None:
                frame_times.append(time.time() - start)
        
        camera.stop_capture()
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time
            print(f"   ✅ Camera FPS: {fps:.1f}")
        else:
            print("   ❌ Camera test failed")
    else:
        print("   ❌ Camera initialization failed")
    
    # Test 3: AI Inference Performance
    print("\n3. AI Inference Performance")
    
    # Generate test images
    test_images = []
    for i in range(10):
        # Create synthetic test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(img)
    
    # Benchmark classification speed
    inference_times = []
    
    for i, test_img in enumerate(test_images):
        print(f"   Testing image {i+1}/10...", end=" ")
        
        start_time = time.time()
        result = classifier.classify_waste(test_img)
        inference_time = time.time() - start_time
        
        inference_times.append(inference_time)
        monitor.log_inference(inference_time)
        
        print(f"{inference_time:.3f}s")
    
    # Calculate statistics
    avg_inference = sum(inference_times) / len(inference_times)
    min_inference = min(inference_times)
    max_inference = max(inference_times)
    
    print(f"\n   📊 Inference Statistics:")
    print(f"   Average: {avg_inference:.3f}s")
    print(f"   Minimum: {min_inference:.3f}s")
    print(f"   Maximum: {max_inference:.3f}s")
    print(f"   Effective FPS: {1.0/avg_inference:.1f}")
    
    # Test 4: Memory Usage
    print("\n4. Memory Usage")
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        print(f"   RAM Usage: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"   Memory Percent: {memory_percent:.1f}%")
        
        # System memory
        system_memory = psutil.virtual_memory()
        print(f"   System RAM: {system_memory.percent:.1f}% used")
        
    except ImportError:
        print("   psutil not available")
    
    # Test 5: CPU Temperature
    print("\n5. System Temperature")
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        print(f"   CPU Temperature: {temp:.1f}°C")
        
        if temp > 70:
            print("   ⚠️  High temperature - consider cooling")
        elif temp > 60:
            print("   ⚠️  Moderate temperature")
        else:
            print("   ✅ Temperature normal")
            
    except Exception as e:
        print(f"   Temperature reading failed: {e}")
    
    # Test 6: System Resources
    print("\n6. System Resources")
    system_stats = monitor.monitor_system()
    if 'error' not in system_stats:
        print(f"   CPU Usage: {system_stats['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {system_stats['memory_percent']:.1f}%")
        print(f"   Temperature: {system_stats['temperature']:.1f}°C")
    
    # Performance Summary
    print("\n" + "="*50)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*50)
    
    # Classification performance rating
    if avg_inference < 1.0:
        perf_rating = "🔥 EXCELLENT"
    elif avg_inference < 2.0:
        perf_rating = "✅ GOOD"
    elif avg_inference < 3.0:
        perf_rating = "⚠️  ACCEPTABLE"
    else:
        perf_rating = "❌ POOR"
    
    print(f"Classification Speed: {perf_rating}")
    print(f"Average Inference: {avg_inference:.3f}s")
    print(f"Target: < 2.0s (✅ {'PASSED' if avg_inference < 2.0 else 'FAILED'})")
    
    # Memory performance
    if 'memory_percent' in system_stats:
        memory_rating = "✅ GOOD" if system_stats['memory_percent'] < 80 else "⚠️  HIGH"
        print(f"Memory Usage: {memory_rating}")
    
    # Temperature performance
    if 'temperature' in system_stats:
        temp_rating = "✅ GOOD" if system_stats['temperature'] < 65 else "⚠️  HIGH"
        print(f"Thermal Performance: {temp_rating}")
    
    print("\n🏁 Benchmark completed!")

if __name__ == "__main__":
    benchmark_system()
```

---

## ✅ Testing & Validation

### Step 1: Create Comprehensive Test Suite

```python
# Create tests/test_rpi_system.py
import unittest
import time
import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rpi_camera_manager import RPiCameraManager
from tflite_classifier import TFLiteWasteClassifier
from hardware_controller import HardwareController
from web_server import app

class TestRPiSystem(unittest.TestCase):
    """Comprehensive test suite for Raspberry Pi deployment"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("Setting up test environment...")
        cls.camera = None
        cls.classifier = None
        cls.hardware = None
    
    def setUp(self):
        """Set up each test"""
        pass
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.camera:
            cls.camera.stop_capture()
        if cls.hardware:
            cls.hardware.cleanup()
    
    def test_01_camera_initialization(self):
        """Test camera initialization and basic functionality"""
        print("\n🧪 Testing camera initialization...")
        
        self.__class__.camera = RPiCameraManager()
        
        # Test initialization
        self.assertTrue(self.__class__.camera.initialize(), "Camera should initialize successfully")
        
        # Test capture
        success = self.__class__.camera.start_capture()
        self.assertTrue(success, "Camera should start capture successfully")
        
        # Test frame capture
        time.sleep(2)  # Allow camera to warm up
        frame = self.__class__.camera.get_frame()
        self.assertIsNotNone(frame, "Should capture frame")
        self.assertEqual(len(frame.shape), 3, "Frame should be 3-dimensional")
        
        # Test high-res capture
        image, timestamp = self.__class__.camera.capture_image()
        self.assertIsNotNone(image, "Should capture high-res image")
        self.assertIsNotNone(timestamp, "Should generate timestamp")
        
        print("✅ Camera tests passed")
    
    def test_02_model_loading(self):
        """Test AI model loading and basic inference"""
        print("\n🧪 Testing AI model loading...")
        
        model_paths = {
            'mobilenet': 'models_rpi/mobilenet_quantized.tflite',
            'coco': 'models_rpi/detect.tflite'
        }
        
        # Check if model files exist
        for name, path in model_paths.items():
            self.assertTrue(os.path.exists(path), f"{name} model file should exist at {path}")
        
        # Initialize classifier
        self.__class__.classifier = TFLiteWasteClassifier(
            mobilenet_path=model_paths['mobilenet'],
            coco_path=model_paths['coco']
        )
        
        # Test model loading
        success = self.__class__.classifier.load_models()
        self.assertTrue(success, "Models should load successfully")
        
        print("✅ Model loading tests passed")
    
    def test_03_classification_accuracy(self):
        """Test classification accuracy with synthetic data"""
        print("\n🧪 Testing classification accuracy...")
        
        self.assertIsNotNone(self.__class__.classifier, "Classifier should be initialized")
        
        # Test with synthetic images
        test_cases = [
            # (image_type, expected_category)
            ("bottle", "DRY WASTE"),
            ("organic", "WET WASTE"),
        ]
        
        for image_type, expected in test_cases:
            # Create synthetic test image
            if image_type == "bottle":
                # Create bottle-like image (mostly blue/clear)
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                test_image[:, :, 2] = 200  # Blue channel
            else:
                # Create organic-like image (mostly green/brown)
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                test_image[:, :, 1] = 150  # Green channel
                test_image[:, :, 0] = 100  # Some blue
            
            # Classify
            result = self.__class__.classifier.classify_waste(test_image)
            
            # Validate result structure
            self.assertIn('waste_category', result)
            self.assertIn('confidence', result)
            self.assertIn('processing_time', result)
            
            # Check processing time
            self.assertLess(result['processing_time'], 5.0, "Processing should be under 5 seconds")
            
            print(f"   {image_type}: {result['waste_category']} (confidence: {result['confidence']:.2f})")
        
        print("✅ Classification tests passed")
    
    def test_04_hardware_components(self):
        """Test hardware components (LEDs, servo, etc.)"""
        print("\n🧪 Testing hardware components...")
        
        try:
            self.__class__.hardware = HardwareController()
            
            # Test LED control
            for color in ['RED', 'GREEN', 'BLUE']:
                print(f"   Testing {color} LED...")
                if color == 'RED':
                    self.__class__.hardware.control_led(self.__class__.hardware.LEDColor.RED, True)
                    time.sleep(0.5)
                    self.__class__.hardware.control_led(self.__class__.hardware.LEDColor.RED, False)
                # Similar for other colors...
            
            # Test servo movement
            print("   Testing servo movement...")
            self.__class__.hardware.move_servo(90)  # Neutral
            time.sleep(1)
            
            # Test distance sensor
            print("   Testing distance sensor...")
            distance = self.__class__.hardware.measure_distance()
            if distance is not None:
                self.assertGreater(distance, 0, "Distance should be positive")
                self.assertLess(distance, 400, "Distance should be reasonable")
            
            print("✅ Hardware tests passed")
            
        except Exception as e:
            print(f"⚠️  Hardware test skipped: {e}")
            # Hardware tests are optional in case of missing components
    
    def test_05_web_interface(self):
        """Test web interface endpoints"""
        print("\n🧪 Testing web interface...")
        
        with app.test_client() as client:
            # Test main page
            response = client.get('/')
            self.assertEqual(response.status_code, 200, "Main page should load")
            
            # Test status endpoint
            response = client.get('/status')
            self.assertEqual(response.status_code, 200, "Status endpoint should work")
            
            data = response.get_json()
            self.assertIn('camera', data, "Status should include camera info")
            self.assertIn('classifier', data, "Status should include classifier info")
            
            # Test Arduino ports endpoint
            response = client.get('/arduino/ports')
            self.assertEqual(response.status_code, 200, "Arduino ports endpoint should work")
            
            print("✅ Web interface tests passed")
    
    def test_06_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n🧪 Testing performance benchmarks...")
        
        if self.__class__.classifier is None:
            self.skipTest("Classifier not initialized")
        
        # Benchmark inference speed
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        inference_times = []
        for i in range(5):
            start_time = time.time()
            result = self.__class__.classifier.classify_waste(test_image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        avg_inference = sum(inference_times) / len(inference_times)
        
        # Performance assertions
        self.assertLess(avg_inference, 5.0, "Average inference should be under 5 seconds")
        
        print(f"   Average inference time: {avg_inference:.3f}s")
        
        # Memory usage test
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.assertLess(memory_mb, 4000, "Memory usage should be under 4GB")
            print(f"   Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            print("   Memory test skipped (psutil not available)")
        
        print("✅ Performance tests passed")
    
    def test_07_integration_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🧪 Testing integration workflow...")
        
        if self.__class__.camera is None or self.__class__.classifier is None:
            self.skipTest("Components not initialized")
        
        # Complete workflow test
        print("   1. Capturing image...")
        image, timestamp = self.__class__.camera.capture_image()
        self.assertIsNotNone(image, "Should capture image")
        
        print("   2. Classifying waste...")
        result = self.__class__.classifier.classify_waste(image)
        self.assertIn('waste_category', result)
        
        print("   3. Hardware response...")
        if self.__class__.hardware:
            self.__class__.hardware.indicate_waste_type(result['waste_category'])
        
        print(f"   Result: {result['waste_category']} (confidence: {result['confidence']:.2f})")
        
        print("✅ Integration workflow tests passed")

def run_validation_suite():
    """Run complete validation suite"""
    print("🚀 Starting Raspberry Pi Validation Suite")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRPiSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED! System ready for deployment.")
    else:
        print(f"\n⚠️  {failures + errors} test(s) failed. Check logs above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
```

### Step 2: Create Deployment Validation Script

```bash
# Create scripts/validate_deployment.sh
#!/bin/bash

echo "🔍 Raspberry Pi 5 Deployment Validation"
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check 1: System Requirements
echo -e "\n1. Checking System Requirements..."

# Check Raspberry Pi model
if grep -q "Raspberry Pi 5" /proc/cpuinfo; then
    print_status 0 "Raspberry Pi 5 detected"
else
    print_status 1 "Not running on Raspberry Pi 5"
fi

# Check RAM
TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
if [ "$TOTAL_RAM" -ge 7 ]; then
    print_status 0 "RAM: ${TOTAL_RAM}GB (sufficient)"
else
    print_status 1 "RAM: ${TOTAL_RAM}GB (8GB recommended)"
fi

# Check storage
AVAILABLE_SPACE=$(df -h / | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "${AVAILABLE_SPACE%.*}" -ge 10 ]; then
    print_status 0 "Storage: ${AVAILABLE_SPACE}G available"
else
    print_status 1 "Storage: ${AVAILABLE_SPACE}G (low space)"
fi

# Check 2: Software Dependencies
echo -e "\n2. Checking Software Dependencies..."

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" = "3.9" ] || [ "$PYTHON_VERSION" = "3.10" ] || [ "$PYTHON_VERSION" = "3.11" ]; then
    print_status 0 "Python $PYTHON_VERSION installed"
else
    print_status 1 "Python version: $PYTHON_VERSION (3.9+ required)"
fi

# Check virtual environment
if [ -d "venv_rpi" ]; then
    print_status 0 "Virtual environment exists"
else
    print_status 1 "Virtual environment not found"
fi

# Check key packages
source venv_rpi/bin/activate 2>/dev/null

PACKAGES=("cv2" "numpy" "tflite_runtime" "flask" "picamera2")
for package in "${PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        print_status 0 "$package package installed"
    else
        print_status 1 "$package package missing"
    fi
done

# Check 3: Hardware Components
echo -e "\n3. Checking Hardware Components..."

# Camera detection
if libcamera-hello --list-cameras 2>/dev/null | grep -q "Available cameras"; then
    print_status 0 "Camera detected"
else
    print_status 1 "Camera not detected"
fi

# GPIO availability
if [ -d "/sys/class/gpio" ]; then
    print_status 0 "GPIO interface available"
else
    print_status 1 "GPIO interface not available"
fi

# Check for I2C/SPI if needed
if [ -e "/dev/i2c-1" ]; then
    print_status 0 "I2C interface available"
else
    print_warning "I2C interface not available (optional)"
fi

# Check 4: Model Files
echo -e "\n4. Checking AI Model Files..."

MODEL_FILES=("models_rpi/mobilenet_quantized.tflite" "models_rpi/detect.tflite")
for model in "${MODEL_FILES[@]}"; do
    if [ -f "$model" ]; then
        SIZE=$(du -h "$model" | cut -f1)
        print_status 0 "$model exists ($SIZE)"
    else
        print_status 1 "$model not found"
    fi
done

# Check 5: Service Configuration
echo -e "\n5. Checking Service Configuration..."

# Check if Nginx is configured
if [ -f "/etc/nginx/sites-available/waste-management" ]; then
    print_status 0 "Nginx configuration exists"
else
    print_status 1 "Nginx configuration missing"
fi

# Check if systemd service exists
if [ -f "/etc/systemd/system/waste-management.service" ]; then
    print_status 0 "Systemd service configured"
else
    print_warning "Systemd service not configured (optional)"
fi

# Check 6: Performance Test
echo -e "\n6. Running Performance Test..."

# Quick inference test
if python3 -c "
import sys
sys.path.append('src')
try:
    from tflite_classifier import TFLiteWasteClassifier
    import numpy as np
    import time
    
    classifier = TFLiteWasteClassifier(
        'models_rpi/mobilenet_quantized.tflite',
        'models_rpi/detect.tflite'
    )
    
    if classifier.load_models():
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        start = time.time()
        result = classifier.classify_waste(test_img)
        inference_time = time.time() - start
        
        print(f'INFERENCE_TIME:{inference_time:.3f}')
        print('SUCCESS')
    else:
        print('FAILED')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null > /tmp/perf_test.out; then
    INFERENCE_TIME=$(grep "INFERENCE_TIME:" /tmp/perf_test.out | cut -d':' -f2)
    if [ -n "$INFERENCE_TIME" ]; then
        if (( $(echo "$INFERENCE_TIME < 5.0" | bc -l) )); then
            print_status 0 "Inference speed: ${INFERENCE_TIME}s (good)"
        else
            print_status 1 "Inference speed: ${INFERENCE_TIME}s (slow)"
        fi
    else
        print_status 1 "Performance test failed"
    fi
else
    print_status 1 "Performance test failed"
fi

rm -f /tmp/perf_test.out

# Check 7: Temperature and Throttling
echo -e "\n7. Checking System Health..."

# CPU temperature
if [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$((TEMP / 1000))
    
    if [ $TEMP_C -lt 65 ]; then
        print_status 0 "CPU temperature: ${TEMP_C}°C (good)"
    elif [ $TEMP_C -lt 75 ]; then
        print_warning "CPU temperature: ${TEMP_C}°C (moderate)"
    else
        print_status 1 "CPU temperature: ${TEMP_C}°C (high)"
    fi
else
    print_warning "Temperature sensor not available"
fi

# Check for throttling
if vcgencmd get_throttled | grep -q "0x0"; then
    print_status 0 "No throttling detected"
else
    THROTTLE_STATE=$(vcgencmd get_throttled)
    print_status 1 "Throttling detected: $THROTTLE_STATE"
fi

# Final Summary
echo -e "\n========================================"
echo "🎯 DEPLOYMENT VALIDATION COMPLETE"
echo "========================================"

# Count results
TOTAL_CHECKS=$(grep -E "(✅|❌)" /tmp/validation.log 2>/dev/null | wc -l || echo "0")
PASSED_CHECKS=$(grep -E "✅" /tmp/validation.log 2>/dev/null | wc -l || echo "0")

if [ "$PASSED_CHECKS" -eq "$TOTAL_CHECKS" ] && [ "$TOTAL_CHECKS" -gt 0 ]; then
    echo -e "${GREEN}🎉 All critical checks passed! System ready for deployment.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some issues detected. Review the output above.${NC}"
    echo -e "${YELLOW}   Passed: $PASSED_CHECKS/$TOTAL_CHECKS checks${NC}"
    exit 1
fi
```

---

## 📦 Deployment Package

### Step 1: Create Automated Installation Script

```bash
# Create install_rpi.sh
#!/bin/bash

set -e  # Exit on any error

echo "🚀 Smart AI Waste Management System - Raspberry Pi 5 Installation"
echo "================================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="/tmp/waste_management_install.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

print_step() {
    echo -e "\n${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if running on Raspberry Pi 5
check_system() {
    print_step "Checking system compatibility..."
    
    if ! grep -q "Raspberry Pi 5" /proc/cpuinfo; then
        print_error "This script is designed for Raspberry Pi 5"
        exit 1
    fi
    
    TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    if [ "$TOTAL_RAM" -lt 4 ]; then
        print_warning "Only ${TOTAL_RAM}GB RAM detected. 8GB recommended."
    fi
    
    print_success "System compatibility verified"
}

# Update system
update_system() {
    print_step "Updating system packages..."
    
    sudo apt update -y
    sudo apt upgrade -y
    
    print_success "System updated"
}

# Install dependencies
install_dependencies() {
    print_step "Installing system dependencies..."
    
    # Essential packages
    sudo apt install -y \
        git curl wget vim htop tree \
        build-essential cmake pkg-config \
        python3-pip python3-venv python3-dev \
        libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
        libopencv-dev python3-opencv \
        nginx \
        bc
    
    # Raspberry Pi specific
    sudo apt install -y \
        python3-picamera2 \
        libcamera-dev libcamera-tools \
        python3-libcamera
    
    print_success "Dependencies installed"
}

# Configure system settings
configure_system() {
    print_step "Configuring system settings..."
    
    # Enable camera
    sudo raspi-config nonint do_camera 0
    
    # Enable I2C and SPI (for sensors)
    sudo raspi-config nonint do_i2c 0
    sudo raspi-config nonint do_spi 0
    
    # GPU memory split
    if ! grep -q "gpu_mem=128" /boot/config.txt; then
        echo 'gpu_mem=128' | sudo tee -a /boot/config.txt
    fi
    
    # Performance settings
    if ! grep -q "arm_boost=1" /boot/config.txt; then
        echo 'arm_boost=1' | sudo tee -a /boot/config.txt
        echo 'over_voltage=2' | sudo tee -a /boot/config.txt
        echo 'arm_freq=2400' | sudo tee -a /boot/config.txt
    fi
    
    print_success "System configured"
}

# Setup project
setup_project() {
    print_step "Setting up project..."
    
    PROJECT_DIR="/home/pi/smart-ai-waste-management"
    
    # Clone repository if it doesn't exist
    if [ ! -d "$PROJECT_DIR" ]; then
        cd /home/pi
        git clone https://github.com/raviramp36/smart-ai-waste-management.git
    fi
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    python3 -m venv venv_rpi
    source venv_rpi/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [ -f "requirements_rpi.txt" ]; then
        pip install -r requirements_rpi.txt
    else
        print_warning "requirements_rpi.txt not found, installing basic packages"
        pip install opencv-python numpy pillow flask flask-cors
        pip install tflite-runtime picamera2 RPi.GPIO pyserial psutil
    fi
    
    # Set permissions
    sudo chown -R pi:pi "$PROJECT_DIR"
    
    print_success "Project setup completed"
}

# Download and setup models
setup_models() {
    print_step "Setting up AI models..."
    
    cd /home/pi/smart-ai-waste-management
    
    # Create models directory
    mkdir -p models_rpi
    
    # Download TensorFlow Lite models
    if [ -f "scripts/convert_models.py" ]; then
        source venv_rpi/bin/activate
        python scripts/convert_models.py
    else
        print_warning "Model conversion script not found"
        
        # Download pre-converted models (fallback)
        cd models_rpi
        
        # Download COCO-SSD Lite model
        if [ ! -f "detect.tflite" ]; then
            wget -q https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
            unzip -q coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
            mv detect.tflite detect.tflite
            rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
        fi
        
        cd ..
    fi
    
    print_success "Models setup completed"
}

# Configure web server
setup_web_server() {
    print_step "Configuring web server..."
    
    # Create Nginx configuration
    sudo tee /etc/nginx/sites-available/waste-management << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /static {
        alias /home/pi/smart-ai-waste-management/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF
    
    # Enable site
    sudo ln -sf /etc/nginx/sites-available/waste-management /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test and start Nginx
    sudo nginx -t
    sudo systemctl enable nginx
    sudo systemctl restart nginx
    
    print_success "Web server configured"
}

# Create systemd service
create_service() {
    print_step "Creating system service..."
    
    sudo tee /etc/systemd/system/waste-management.service << 'EOF'
[Unit]
Description=Smart AI Waste Management System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-ai-waste-management
Environment=PATH=/home/pi/smart-ai-waste-management/venv_rpi/bin
ExecStart=/home/pi/smart-ai-waste-management/venv_rpi/bin/python src/web_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable service
    sudo systemctl daemon-reload
    sudo systemctl enable waste-management
    
    print_success "System service created"
}

# Create startup scripts
create_scripts() {
    print_step "Creating management scripts..."
    
    # Start script
    tee /home/pi/start_waste_management.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Smart AI Waste Management System..."

cd /home/pi/smart-ai-waste-management
source venv_rpi/bin/activate

echo "Starting web server..."
sudo systemctl start waste-management
sudo systemctl start nginx

echo "✅ System started!"
echo "🌐 Access the interface at: http://$(hostname -I | awk '{print $1}')"
EOF

    # Stop script
    tee /home/pi/stop_waste_management.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Smart AI Waste Management System..."

sudo systemctl stop waste-management
echo "✅ System stopped!"
EOF

    # Status script
    tee /home/pi/status_waste_management.sh << 'EOF'
#!/bin/bash
echo "📊 Smart AI Waste Management System Status"
echo "=========================================="

echo -n "Web Server: "
if systemctl is-active --quiet waste-management; then
    echo "✅ Running"
else
    echo "❌ Stopped"
fi

echo -n "Nginx: "
if systemctl is-active --quiet nginx; then
    echo "✅ Running"
else
    echo "❌ Stopped"
fi

echo -n "Camera: "
if libcamera-hello --list-cameras 2>/dev/null | grep -q "Available cameras"; then
    echo "✅ Detected"
else
    echo "❌ Not detected"
fi

echo -n "Temperature: "
if [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$((TEMP / 1000))
    echo "${TEMP_C}°C"
else
    echo "Unknown"
fi

echo ""
echo "🌐 Web Interface: http://$(hostname -I | awk '{print $1}')"
EOF

    # Make scripts executable
    chmod +x /home/pi/*_waste_management.sh
    
    print_success "Management scripts created"
}

# Run tests
run_tests() {
    print_step "Running validation tests..."
    
    cd /home/pi/smart-ai-waste-management
    
    if [ -f "scripts/validate_deployment.sh" ]; then
        chmod +x scripts/validate_deployment.sh
        bash scripts/validate_deployment.sh
    else
        print_warning "Validation script not found, skipping tests"
    fi
}

# Main installation process
main() {
    echo "Starting installation process..."
    echo "This may take 30-60 minutes depending on your internet connection."
    echo ""
    
    check_system
    update_system
    install_dependencies
    configure_system
    setup_project
    setup_models
    setup_web_server
    create_service
    create_scripts
    run_tests
    
    echo ""
    echo "================================================================="
    echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
    echo "================================================================="
    echo ""
    echo "Next steps:"
    echo "1. Reboot your Raspberry Pi: sudo reboot"
    echo "2. After reboot, start the system: ./start_waste_management.sh"
    echo "3. Access the web interface at: http://$(hostname -I | awk '{print $1}')"
    echo ""
    echo "Management commands:"
    echo "  Start:  ./start_waste_management.sh"
    echo "  Stop:   ./stop_waste_management.sh"
    echo "  Status: ./status_waste_management.sh"
    echo ""
    echo "📚 View the full guide: less RPi5_DEPLOYMENT_GUIDE.md"
    echo ""
}

# Run installation
main "$@"
```

### Step 2: Create Quick Setup Script

```bash
# Create quick_setup.sh
#!/bin/bash

echo "⚡ Smart AI Waste Management - Quick Setup"
echo "=========================================="

# One-liner installer
curl -fsSL https://raw.githubusercontent.com/raviramp36/smart-ai-waste-management/main/install_rpi.sh | bash

echo ""
echo "🚀 Quick setup completed!"
echo "Run 'sudo reboot' and then './start_waste_management.sh'"
```

### Step 3: Create Configuration Management

```python
# Create src/config_manager.py
import json
import os
from typing import Dict, Any

class ConfigManager:
    """
    Configuration management for Raspberry Pi deployment
    Handles system settings, camera parameters, and model configurations
    """
    
    def __init__(self, config_file="config/rpi_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "camera": {
                "resolution": [1920, 1080],
                "fps": 30,
                "auto_focus": True,
                "brightness": 0.0,
                "contrast": 1.2,
                "saturation": 1.1
            },
            "ai": {
                "mobilenet_path": "models_rpi/mobilenet_quantized.tflite",
                "coco_path": "models_rpi/detect.tflite",
                "num_threads": 4,
                "confidence_threshold": 0.5
            },
            "hardware": {
                "led_pins": {
                    "red": 18,
                    "green": 19,
                    "blue": 20
                },
                "servo_pin": 21,
                "buzzer_pin": 22,
                "ultrasonic_pins": {
                    "trigger": 23,
                    "echo": 24
                }
            },
            "arduino": {
                "port": "/dev/ttyUSB0",
                "baud_rate": 9600,
                "timeout": 1,
                "commands": {
                    "wet_waste": "w",
                    "dry_waste": "d",
                    "reset": "r",
                    "status": "s"
                }
            },
            "web": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            },
            "performance": {
                "max_inference_time": 5.0,
                "memory_limit_mb": 4000,
                "temperature_limit_c": 75
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                return self.merge_configs(default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
                return default_config
        else:
            # Save default config
            self.save_config(default_config)
            return default_config
    
    def merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge configurations"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'camera.resolution')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()
    
    def update_camera_config(self, **kwargs):
        """Update camera configuration"""
        for key, value in kwargs.items():
            self.set(f"camera.{key}", value)
    
    def update_ai_config(self, **kwargs):
        """Update AI configuration"""
        for key, value in kwargs.items():
            self.set(f"ai.{key}", value)
    
    def update_hardware_config(self, **kwargs):
        """Update hardware configuration"""
        for key, value in kwargs.items():
            self.set(f"hardware.{key}", value)
    
    def get_camera_config(self) -> Dict:
        """Get camera configuration"""
        return self.get('camera', {})
    
    def get_ai_config(self) -> Dict:
        """Get AI configuration"""
        return self.get('ai', {})
    
    def get_hardware_config(self) -> Dict:
        """Get hardware configuration"""
        return self.get('hardware', {})
    
    def get_arduino_config(self) -> Dict:
        """Get Arduino configuration"""
        return self.get('arduino', {})
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate camera settings
        resolution = self.get('camera.resolution')
        if not isinstance(resolution, list) or len(resolution) != 2:
            errors.append("Invalid camera resolution format")
        
        # Validate file paths
        ai_config = self.get_ai_config()
        for path_key in ['mobilenet_path', 'coco_path']:
            path = ai_config.get(path_key)
            if path and not os.path.exists(path):
                errors.append(f"Model file not found: {path}")
        
        # Validate hardware pins
        hardware_config = self.get_hardware_config()
        led_pins = hardware_config.get('led_pins', {})
        all_pins = list(led_pins.values())
        all_pins.extend([
            hardware_config.get('servo_pin'),
            hardware_config.get('buzzer_pin')
        ])
        
        # Check for pin conflicts
        used_pins = [p for p in all_pins if p is not None]
        if len(used_pins) != len(set(used_pins)):
            errors.append("Duplicate GPIO pin assignments detected")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Create configuration file template
def create_config_template():
    """Create a configuration template file"""
    config_manager = ConfigManager()
    print("Configuration template created at: config/rpi_config.json")
    print("Edit this file to customize your deployment.")

if __name__ == "__main__":
    create_config_template()
```

---

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### 1. Camera Issues

**Problem**: Camera not detected
```bash
# Check camera connection
libcamera-hello --list-cameras

# Enable camera in raspi-config
sudo raspi-config
# Interface Options → Camera → Enable

# Check config.txt
grep camera /boot/config.txt
```

**Problem**: Poor image quality
```python
# Adjust camera settings in config
{
    "camera": {
        "brightness": 0.1,    # Increase for darker conditions
        "contrast": 1.3,      # Enhance contrast
        "saturation": 1.2     # Boost colors
    }
}
```

#### 2. AI Model Issues

**Problem**: Models not loading
```bash
# Check model files exist
ls -la models_rpi/
file models_rpi/*.tflite

# Test model loading
python3 -c "
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter('models_rpi/detect.tflite')
print('Model loaded successfully')
"
```

**Problem**: Slow inference
```python
# Optimize model settings
{
    "ai": {
        "num_threads": 4,           # Use all CPU cores
        "confidence_threshold": 0.6  # Higher threshold for faster processing
    }
}
```

#### 3. Hardware Issues

**Problem**: GPIO permissions
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi

# Check GPIO permissions
ls -la /dev/gpiomem
```

**Problem**: Arduino not connecting
```bash
# Check USB devices
lsusb
ls /dev/ttyUSB* /dev/ttyACM*

# Check permissions
sudo usermod -a -G dialout pi

# Test connection manually
python3 -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
print('Arduino connected')
ser.close()
"
```

#### 4. Performance Issues

**Problem**: High CPU temperature
```bash
# Check temperature
vcgencmd measure_temp

# Add cooling
# Install heatsink or fan

# Check throttling
vcgencmd get_throttled
```

**Problem**: Memory issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Increase swap
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 5. Web Interface Issues

**Problem**: Can't access web interface
```bash
# Check service status
sudo systemctl status waste-management
sudo systemctl status nginx

# Check ports
netstat -tlnp | grep :80
netstat -tlnp | grep :5000

# Check logs
journalctl -u waste-management -f
tail -f /var/log/nginx/error.log
```

**Problem**: HTTPS required for camera
```bash
# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/waste-management.key \
    -out /etc/ssl/certs/waste-management.crt

# Update Nginx config for HTTPS
# (Add SSL configuration)
```

---

## 🔄 Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Checks
```bash
# Check system status
./status_waste_management.sh

# Monitor temperature
vcgencmd measure_temp

# Check disk space
df -h
```

#### Weekly Maintenance
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Clean logs
sudo journalctl --vacuum-time=7d

# Check model performance
python3 scripts/benchmark_performance.py
```

#### Monthly Tasks
```bash
# Full system backup
sudo dd if=/dev/mmcblk0 of=/path/to/backup.img bs=4M status=progress

# Update AI models (if available)
python3 scripts/update_models.py

# Performance audit
python3 tests/test_rpi_system.py
```

### Update Procedures

#### Software Updates
```bash
# Update application
cd /home/pi/smart-ai-waste-management
git pull origin main
source venv_rpi/bin/activate
pip install -r requirements_rpi.txt --upgrade

# Restart services
sudo systemctl restart waste-management
```

#### Model Updates
```bash
# Download new models
python3 scripts/convert_models.py

# Test new models
python3 scripts/benchmark_performance.py

# Deploy if tests pass
sudo systemctl restart waste-management
```

---

**🎉 Deployment Guide Complete!**

This comprehensive guide covers everything needed to deploy the Smart AI Waste Management System on Raspberry Pi 5. Follow the steps sequentially for a successful deployment.

**Quick Start**: Run `curl -fsSL https://raw.githubusercontent.com/your-repo/install_rpi.sh | bash` for automated installation.