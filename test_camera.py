#!/usr/bin/env python3
"""
Simple camera test script for Mac
"""
import cv2
import sys

def test_camera():
    """Test camera functionality"""
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Test frame capture
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        return False
    
    print(f"Camera working! Frame shape: {frame.shape}")
    
    # Show frame for 2 seconds
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(2000)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    if test_camera():
        print("Camera test passed!")
    else:
        print("Camera test failed!")
        sys.exit(1)