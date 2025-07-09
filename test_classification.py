#!/usr/bin/env python3
"""
Test waste classification without camera
"""
import cv2
import numpy as np
from waste_classifier import WasteClassifier
from data_manager import DataManager
import os

def create_test_images():
    """Create test images for classification"""
    # Create sample images
    test_images = {}
    
    # Green/organic looking image (wet waste)
    green_img = np.zeros((224, 224, 3), dtype=np.uint8)
    green_img[:, :, 1] = 150  # Green channel
    green_img[:, :, 0] = 50   # Add some blue
    # Add some texture
    for i in range(0, 224, 20):
        cv2.line(green_img, (i, 0), (i, 224), (0, 100, 0), 2)
    test_images['wet_sample'] = green_img
    
    # Gray/white image (dry waste)
    gray_img = np.full((224, 224, 3), 200, dtype=np.uint8)
    # Add some geometric patterns
    cv2.rectangle(gray_img, (50, 50), (150, 150), (150, 150, 150), -1)
    cv2.circle(gray_img, (112, 112), 30, (100, 100, 100), 3)
    test_images['dry_sample'] = gray_img
    
    # Brown/organic image (wet waste)
    brown_img = np.zeros((224, 224, 3), dtype=np.uint8)
    brown_img[:, :, 2] = 139  # Red
    brown_img[:, :, 1] = 69   # Green
    brown_img[:, :, 0] = 19   # Blue
    # Add organic-looking texture
    for i in range(100):
        x, y = np.random.randint(0, 224, 2)
        cv2.circle(brown_img, (x, y), np.random.randint(2, 8), (100, 50, 20), -1)
    test_images['wet_organic'] = brown_img
    
    # Blue/plastic image (dry waste)
    blue_img = np.zeros((224, 224, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 200  # Blue
    blue_img[:, :, 1] = 50   # Green
    blue_img[:, :, 2] = 50   # Red
    # Add plastic-like smooth texture
    cv2.rectangle(blue_img, (30, 30), (194, 194), (150, 30, 30), 3)
    test_images['dry_plastic'] = blue_img
    
    return test_images

def test_classification():
    """Test classification system"""
    print("Testing Waste Classification System")
    print("=" * 40)
    
    # Initialize components
    classifier = WasteClassifier()
    data_manager = DataManager()
    
    # Create test images
    test_images = create_test_images()
    
    # Test classification
    results = []
    
    for name, image in test_images.items():
        print(f"\\nTesting {name}...")
        
        # Classify image
        waste_type, confidence, processing_time = classifier.classify_waste(image)
        
        if waste_type:
            print(f"  Result: {waste_type}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Processing Time: {processing_time:.3f}s")
            
            # Save result
            results.append({
                'name': name,
                'waste_type': waste_type,
                'confidence': confidence,
                'processing_time': processing_time
            })
            
            # Save to database
            data_manager.save_classification(waste_type, confidence, None, processing_time)
            
        else:
            print(f"  Classification failed")
    
    print(f"\\nClassification Summary:")
    print("-" * 25)
    for result in results:
        print(f"{result['name']}: {result['waste_type']} ({result['confidence']:.1%})")
    
    # Test data manager
    print(f"\\nDatabase Statistics:")
    print("-" * 20)
    stats = data_manager.get_classification_stats()
    for waste_type, data in stats.items():
        print(f"{waste_type}: {data['count']} classifications, {data['avg_confidence']:.1%} avg confidence")
    
    # Test recent classifications
    print(f"\\nRecent Classifications:")
    print("-" * 22)
    recent = data_manager.get_recent_classifications(5)
    for timestamp, waste_type, confidence, proc_time in recent:
        print(f"{timestamp}: {waste_type} ({confidence:.1%})")
    
    return True

if __name__ == "__main__":
    try:
        test_classification()
        print("\\n✅ Classification test completed successfully!")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()