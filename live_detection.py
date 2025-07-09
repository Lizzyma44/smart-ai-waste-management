#!/usr/bin/env python3
"""
Live waste detection using webcam
"""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from datetime import datetime

from waste_classifier import WasteClassifier
from data_manager import DataManager

class LiveWasteDetection:
    """
    Live waste detection with webcam
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Live Waste Detection System")
        self.root.geometry("900x700")
        
        # Initialize components
        self.classifier = WasteClassifier()
        self.data_manager = DataManager()
        
        # Camera variables
        self.cap = None
        self.running = False
        self.current_frame = None
        self.last_classification_time = 0
        self.auto_classify = True
        
        # Create UI
        self._create_ui()
        
        # Try to start camera
        self._initialize_camera()
    
    def _create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Live Waste Detection System", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding="10")
        camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera display - larger size for better visibility
        self.camera_label = ttk.Label(camera_frame, text="Starting camera...", 
                                     anchor="center", font=("Arial", 14))
        self.camera_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(0, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera controls
        cam_controls = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        cam_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_btn = ttk.Button(cam_controls, text="Start Camera", command=self._start_camera)
        self.start_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.stop_btn = ttk.Button(cam_controls, text="Stop Camera", command=self._stop_camera, state="disabled")
        self.stop_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Classification controls
        class_controls = ttk.LabelFrame(control_frame, text="Classification", padding="5")
        class_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.auto_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(class_controls, text="Auto Classify", 
                                   variable=self.auto_var, command=self._toggle_auto)
        auto_check.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.classify_now_btn = ttk.Button(class_controls, text="Classify Now", 
                                         command=self._classify_now, state="disabled")
        self.classify_now_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.save_btn = ttk.Button(class_controls, text="Save Frame", 
                                 command=self._save_frame, state="disabled")
        self.save_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="5")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_btn = ttk.Button(stats_frame, text="View Statistics", command=self._show_stats)
        self.stats_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.clear_btn = ttk.Button(stats_frame, text="Clear History", command=self._clear_history)
        self.clear_btn.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure control frame
        control_frame.columnconfigure(0, weight=1)
        cam_controls.columnconfigure(0, weight=1)
        class_controls.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Live Classification Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Results display with larger, more visible text
        self.result_label = ttk.Label(results_frame, text="No classification yet", 
                                    font=("Arial", 16, "bold"))
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: --", 
                                        font=("Arial", 12))
        self.confidence_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        self.time_label = ttk.Label(results_frame, text="Processing: --", 
                                  font=("Arial", 12))
        self.time_label.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        # Status and count
        self.status_label = ttk.Label(results_frame, text="Ready", 
                                    font=("Arial", 10), foreground="blue")
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.count_label = ttk.Label(results_frame, text="Classifications: 0", 
                                   font=("Arial", 10))
        self.count_label.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        self.fps_label = ttk.Label(results_frame, text="FPS: --", 
                                 font=("Arial", 10))
        self.fps_label.grid(row=1, column=2, sticky=(tk.W, tk.E))
        
        # Configure results frame
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        
        # Initialize counters
        self.classification_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def _initialize_camera(self):
        """Initialize camera with better error handling"""
        try:
            # Try different camera indices and backends
            camera_configs = [
                (0, cv2.CAP_AVFOUNDATION),
                (1, cv2.CAP_AVFOUNDATION),
                (0, cv2.CAP_ANY),
                (1, cv2.CAP_ANY),
                (0, None),
                (1, None)
            ]
            
            for cam_idx, backend in camera_configs:
                try:
                    if backend is not None:
                        cap = cv2.VideoCapture(cam_idx, backend)
                    else:
                        cap = cv2.VideoCapture(cam_idx)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            self.cap = cap
                            self._setup_camera()
                            self.status_label.config(text="Camera ready - Click Start Camera", foreground="green")
                            return
                        else:
                            cap.release()
                except:
                    continue
            
            # If we get here, no camera worked
            self.status_label.config(text="No camera found - Check permissions", foreground="red")
            messagebox.showwarning("Camera Error", 
                                 "Could not access camera. Please check:\\n" +
                                 "1. Camera permissions in System Preferences\\n" +
                                 "2. Camera is not in use by another app\\n" +
                                 "3. Camera is properly connected")
            
        except Exception as e:
            self.status_label.config(text=f"Camera error: {str(e)}", foreground="red")
    
    def _setup_camera(self):
        """Setup camera properties"""
        if self.cap:
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
    
    def _start_camera(self):
        """Start camera capture"""
        if not self.cap:
            self._initialize_camera()
            if not self.cap:
                return
        
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.classify_now_btn.config(state="normal")
        self.save_btn.config(state="normal")
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.status_label.config(text="Camera started - Live detection active", foreground="green")
    
    def _stop_camera(self):
        """Stop camera capture"""
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.classify_now_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        
        self.status_label.config(text="Camera stopped", foreground="blue")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Update FPS counter
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Update display
                self._update_display(frame)
                
                # Auto classify if enabled
                if self.auto_var.get():
                    current_time = time.time()
                    if current_time - self.last_classification_time >= 3.0:  # Every 3 seconds
                        self._classify_frame(frame)
                        self.last_classification_time = current_time
            
            time.sleep(0.033)  # ~30 FPS
    
    def _update_display(self, frame):
        """Update camera display"""
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (500, 375))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display in main thread
            self.root.after(0, lambda: self._update_camera_display(photo))
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def _update_camera_display(self, photo):
        """Update camera display widget"""
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo  # Keep reference
    
    def _classify_now(self):
        """Manual classification trigger"""
        if self.current_frame is not None:
            self._classify_frame(self.current_frame)
    
    def _classify_frame(self, frame):
        """Classify current frame"""
        try:
            # Perform classification
            waste_type, confidence, processing_time = self.classifier.classify_waste(frame)
            
            if waste_type:
                # Update classification count
                self.classification_count += 1
                
                # Update results display
                self._update_results(waste_type, confidence, processing_time)
                
                # Save to database
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = None
                
                if self.data_manager.get_config('classification').get('save_images', True):
                    image_path = self.data_manager.save_image(frame, timestamp, waste_type)
                
                self.data_manager.save_classification(waste_type, confidence, image_path, processing_time)
                
                # Update count display
                self.root.after(0, lambda: self.count_label.config(text=f"Classifications: {self.classification_count}"))
                
        except Exception as e:
            print(f"Classification error: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"Classification error: {e}", foreground="red"))
    
    def _update_results(self, waste_type, confidence, processing_time):
        """Update classification results display"""
        def update():
            # Color coding for waste types
            if waste_type == "Dry Waste":
                color = "#2E7D32"  # Green
                bg_color = "#E8F5E8"
            else:  # Wet Waste
                color = "#8D6E63"  # Brown
                bg_color = "#F3E5AB"
            
            # Update result with color and background
            self.result_label.config(text=f"🗑️ {waste_type}", foreground=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            self.time_label.config(text=f"Processing: {processing_time:.3f}s")
            self.status_label.config(text=f"✅ Classified as {waste_type}", foreground=color)
        
        self.root.after(0, update)
    
    def _toggle_auto(self):
        """Toggle auto classification"""
        self.auto_classify = self.auto_var.get()
        status = "enabled" if self.auto_classify else "disabled"
        self.status_label.config(text=f"Auto classification {status}", foreground="blue")
    
    def _save_frame(self):
        """Save current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_capture_{timestamp}.jpg"
            filepath = f"captured_images/{filename}"
            
            cv2.imwrite(filepath, self.current_frame)
            messagebox.showinfo("Success", f"Frame saved: {filename}")
    
    def _show_stats(self):
        """Show statistics window"""
        stats = self.data_manager.get_classification_stats()
        recent = self.data_manager.get_recent_classifications(15)
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Live Detection Statistics")
        stats_window.geometry("500x400")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=("Courier", 10))
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Add statistics content
        stats_text.insert(tk.END, "📊 LIVE DETECTION STATISTICS\\n")
        stats_text.insert(tk.END, "=" * 50 + "\\n\\n")
        
        stats_text.insert(tk.END, f"Total Classifications: {self.classification_count}\\n")
        stats_text.insert(tk.END, f"Session Duration: {time.time() - self.fps_start_time:.1f}s\\n\\n")
        
        stats_text.insert(tk.END, "Classification Breakdown:\\n")
        stats_text.insert(tk.END, "-" * 30 + "\\n")
        
        for waste_type, data in stats.items():
            stats_text.insert(tk.END, f"{waste_type}:\\n")
            stats_text.insert(tk.END, f"  Count: {data['count']}\\n")
            stats_text.insert(tk.END, f"  Avg Confidence: {data['avg_confidence']:.1%}\\n\\n")
        
        # Recent classifications tab
        recent_frame = ttk.Frame(notebook)
        notebook.add(recent_frame, text="Recent Classifications")
        
        recent_text = tk.Text(recent_frame, wrap=tk.WORD, font=("Courier", 10))
        recent_text.pack(fill=tk.BOTH, expand=True)
        
        recent_text.insert(tk.END, "🕐 RECENT CLASSIFICATIONS\\n")
        recent_text.insert(tk.END, "=" * 40 + "\\n\\n")
        
        for timestamp, waste_type, confidence, proc_time in recent:
            recent_text.insert(tk.END, f"{timestamp}\\n")
            recent_text.insert(tk.END, f"  Type: {waste_type}\\n")
            recent_text.insert(tk.END, f"  Confidence: {confidence:.1%}\\n")
            recent_text.insert(tk.END, f"  Processing: {proc_time:.3f}s\\n\\n")
        
        # Make text read-only
        stats_text.config(state=tk.DISABLED)
        recent_text.config(state=tk.DISABLED)
    
    def _clear_history(self):
        """Clear classification history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all classification history?"):
            # This would require a method in data_manager to clear the database
            self.classification_count = 0
            self.count_label.config(text="Classifications: 0")
            self.status_label.config(text="History cleared", foreground="blue")
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = LiveWasteDetection(root)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start application
    root.mainloop()

if __name__ == "__main__":
    main()