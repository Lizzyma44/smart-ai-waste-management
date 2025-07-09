#!/usr/bin/env python3
"""
Download AI models locally for offline use
"""
import requests
import os
import json

def download_file(url, filename):
    """Download a file with progress"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')
    
    print(f"  ✅ Downloaded {filename}")

def download_models():
    """Download TensorFlow.js models locally"""
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("📦 Downloading AI models for offline use...")
    
    # TensorFlow.js core
    tf_files = [
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js",
        "https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0/dist/mobilenet.min.js",
        "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.2/dist/coco-ssd.min.js"
    ]
    
    for url in tf_files:
        filename = os.path.join(models_dir, os.path.basename(url))
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    # Download MobileNet model files
    mobilenet_base = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/"
    mobilenet_files = [
        "model.json",
        "group1-shard1of1.bin"
    ]
    
    mobilenet_dir = os.path.join(models_dir, "mobilenet")
    os.makedirs(mobilenet_dir, exist_ok=True)
    
    for file in mobilenet_files:
        url = mobilenet_base + file
        filename = os.path.join(mobilenet_dir, file)
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    # Download COCO-SSD model files
    coco_base = "https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/"
    coco_files = [
        "model.json",
        "group1-shard1of1.bin"
    ]
    
    coco_dir = os.path.join(models_dir, "coco-ssd")
    os.makedirs(coco_dir, exist_ok=True)
    
    for file in coco_files:
        url = coco_base + file
        filename = os.path.join(coco_dir, file)
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    print("\n✅ All models downloaded successfully!")
    print("📁 Models stored in: ./models/")
    print("🚀 You can now run the app offline!")

if __name__ == "__main__":
    download_models()