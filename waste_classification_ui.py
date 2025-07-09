import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

from camera_manager import CameraManager
from waste_classifier import WasteClassifier
from data_manager import DataManager

class WasteClassificationUI:
    """
    Main UI application for waste classification system
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Waste Classification System")
        self.root.geometry("800x600")
        
        # Initialize components
        self.camera_manager = CameraManager()
        self.classifier = WasteClassifier()
        self.data_manager = DataManager()
        
        # UI state variables
        self.is_running = False
        self.auto_classify = True
        self.current_image = None
        self.classification_result = None
        
        # Create UI
        self._create_ui()
        
        # Start camera
        self._start_camera()
    
    def _create_ui(self):
        """Create main user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Smart Waste Classification System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Camera feed frame
        camera_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding="5")
        camera_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Starting camera...", anchor="center")
        self.camera_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(0, weight=1)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        
        # Classification button
        self.classify_btn = ttk.Button(control_frame, text="Classify Waste", 
                                      command=self._manual_classify, state="disabled")
        self.classify_btn.grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        
        # Auto classify toggle
        self.auto_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(control_frame, text="Auto Classify", 
                                   variable=self.auto_var, command=self._toggle_auto_classify)
        auto_check.grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        
        # Save image button
        self.save_btn = ttk.Button(control_frame, text="Save Image", 
                                  command=self._save_current_image, state="disabled")
        self.save_btn.grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        
        # Statistics button
        stats_btn = ttk.Button(control_frame, text="View Statistics", 
                              command=self._show_statistics)
        stats_btn.grid(row=3, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Classification result display
        self.result_label = ttk.Label(results_frame, text="No classification yet", 
                                    font=("Arial", 12, "bold"))
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Confidence display
        self.confidence_label = ttk.Label(results_frame, text="Confidence: --")
        self.confidence_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Processing time display
        self.time_label = ttk.Label(results_frame, text="Processing time: --")
        self.time_label.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        # Configure result frame columns
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Starting system...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Configure control frame
        control_frame.columnconfigure(0, weight=1)
    
    def _start_camera(self):
        """Start camera capture"""
        if self.camera_manager.start_capture():
            self.is_running = True
            self.classify_btn.config(state="normal")
            self.save_btn.config(state="normal")
            self.status_var.set("Camera started successfully")
            
            # Start UI update thread
            self.update_thread = threading.Thread(target=self._update_ui_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            
        else:
            messagebox.showerror("Error", "Failed to start camera")
            self.status_var.set("Camera initialization failed")
    
    def _update_ui_loop(self):
        """Update UI in separate thread"""
        while self.is_running:
            try:
                # Get current frame
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    self.current_image = frame
                    
                    # Update camera display
                    self._update_camera_display(frame)
                    
                    # Auto classify if enabled
                    if self.auto_var.get():
                        self._auto_classify(frame)
                
                time.sleep(0.1)  # 10 FPS UI update
                
            except Exception as e:
                print(f"UI update error: {e}")
                time.sleep(1)
    
    def _update_camera_display(self, frame):
        """Update camera display with current frame"""
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (400, 300))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display in main thread
            self.root.after(0, lambda: self._update_camera_label(photo))
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def _update_camera_label(self, photo):
        """Update camera label with new photo"""
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo  # Keep reference
    
    def _auto_classify(self, frame):
        """Perform automatic classification"""
        if not hasattr(self, 'last_classification_time'):
            self.last_classification_time = 0
        
        current_time = time.time()
        if current_time - self.last_classification_time > 2:  # Classify every 2 seconds
            self._classify_image(frame)
            self.last_classification_time = current_time
    
    def _manual_classify(self):
        """Manual classification trigger"""
        if self.current_image is not None:
            self._classify_image(self.current_image)
    
    def _classify_image(self, image):
        """Classify waste image and update results"""
        try:
            # Perform classification
            waste_type, confidence, processing_time = self.classifier.classify_waste(image)
            
            if waste_type:
                # Update UI
                self._update_results(waste_type, confidence, processing_time)
                
                # Save to database
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = None
                
                if self.data_manager.get_config('classification').get('save_images', True):
                    image_path = self.data_manager.save_image(image, timestamp, waste_type)
                
                self.data_manager.save_classification(waste_type, confidence, image_path, processing_time)
                
                # Update status
                self.root.after(0, lambda: self.status_var.set(f"Classified as {waste_type} ({confidence:.1%})"))
                
        except Exception as e:
            print(f"Classification error: {e}")
            self.root.after(0, lambda: self.status_var.set("Classification error"))
    
    def _update_results(self, waste_type, confidence, processing_time):
        """Update classification results display"""
        def update():
            # Update result label with color coding
            color = "#2E7D32" if waste_type == "Dry Waste" else "#8D6E63"  # Green for dry, brown for wet
            self.result_label.config(text=f"Result: {waste_type}", foreground=color)
            
            # Update confidence
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Update processing time
            self.time_label.config(text=f"Processing: {processing_time:.3f}s")
        
        self.root.after(0, update)
    
    def _toggle_auto_classify(self):
        """Toggle auto classification"""
        self.auto_classify = self.auto_var.get()
        status = "enabled" if self.auto_classify else "disabled"
        self.status_var.set(f"Auto classification {status}")
    
    def _save_current_image(self):
        """Save current image manually"""
        if self.current_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.data_manager.save_image(self.current_image, timestamp, "manual")
            if image_path:
                messagebox.showinfo("Success", f"Image saved to: {image_path}")
            else:
                messagebox.showerror("Error", "Failed to save image")
    
    def _show_statistics(self):
        """Show classification statistics"""
        stats = self.data_manager.get_classification_stats()
        recent = self.data_manager.get_recent_classifications(5)
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Classification Statistics")
        stats_window.geometry("400x300")
        
        # Statistics display
        stats_text = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Add statistics content
        stats_text.insert(tk.END, "Classification Statistics (Last 7 days)\\n")
        stats_text.insert(tk.END, "=" * 40 + "\\n\\n")
        
        for waste_type, data in stats.items():
            stats_text.insert(tk.END, f"{waste_type}:\\n")
            stats_text.insert(tk.END, f"  Count: {data['count']}\\n")
            stats_text.insert(tk.END, f"  Avg Confidence: {data['avg_confidence']:.1%}\\n\\n")
        
        stats_text.insert(tk.END, "Recent Classifications:\\n")
        stats_text.insert(tk.END, "-" * 20 + "\\n")
        
        for timestamp, waste_type, confidence, proc_time in recent:
            stats_text.insert(tk.END, f"{timestamp}: {waste_type} ({confidence:.1%})\\n")
        
        stats_text.config(state=tk.DISABLED)
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.camera_manager.stop_capture()
        self.data_manager.log_system_event("INFO", "System shutdown", "UI")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = WasteClassificationUI(root)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start application
    root.mainloop()

if __name__ == "__main__":
    main()