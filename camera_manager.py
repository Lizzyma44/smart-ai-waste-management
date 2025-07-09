import cv2
import numpy as np
from datetime import datetime
import threading
import time

class CameraManager:
    """
    Camera management class for Mac webcam integration
    Replaces Pi camera functionality for testing
    """
    
    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize camera connection"""
        try:
            # Try different camera backends for Mac compatibility
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                    if self.cap.isOpened():
                        # Test capture
                        ret, frame = self.cap.read()
                        if ret:
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    else:
                        self.cap = None
                except:
                    self.cap = None
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]}")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def start_capture(self):
        """Start continuous frame capture in separate thread"""
        if not self.cap:
            if not self.initialize():
                return False
                
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def capture_image(self):
        """Capture single image for classification"""
        frame = self.get_frame()
        if frame is not None:
            # Add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return frame, timestamp
        return None, None
    
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.stop_capture()