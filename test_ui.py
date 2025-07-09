#!/usr/bin/env python3
"""
Test UI without camera for Mac testing
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from waste_classifier import WasteClassifier
from data_manager import DataManager
import os

class TestWasteClassificationUI:
    """
    Test UI for waste classification without camera
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classification System - Test Mode")
        self.root.geometry("600x500")
        
        # Initialize components
        self.classifier = WasteClassifier()
        self.data_manager = DataManager()
        
        # Current image
        self.current_image = None
        
        # Create UI
        self._create_ui()
        
        # Load sample images
        self._load_sample_images()
    
    def _create_ui(self):
        """Create test UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Waste Classification System - Test Mode", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Test Image", padding="5")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Image display
        self.image_label = ttk.Label(image_frame, text="No image loaded", 
                                    anchor="center", background="white")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Load image button
        load_btn = ttk.Button(controls_frame, text="Load Image", command=self._load_image)
        load_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Sample images dropdown
        ttk.Label(controls_frame, text="Sample:").grid(row=0, column=1, padx=(10, 5))
        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(controls_frame, textvariable=self.sample_var, 
                                        values=[], state="readonly", width=15)
        self.sample_combo.grid(row=0, column=2, padx=(0, 5))
        self.sample_combo.bind('<<ComboboxSelected>>', self._load_sample)
        
        # Classify button
        self.classify_btn = ttk.Button(controls_frame, text="Classify", 
                                     command=self._classify_current, state="disabled")
        self.classify_btn.grid(row=0, column=3, padx=(5, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="5")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results display
        self.result_label = ttk.Label(results_frame, text="No classification yet", 
                                    font=("Arial", 12, "bold"))
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: --")
        self.confidence_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        self.time_label = ttk.Label(results_frame, text="Processing time: --")
        self.time_label.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        
        # Statistics button
        stats_btn = ttk.Button(main_frame, text="View Statistics", command=self._show_stats)
        stats_btn.grid(row=4, column=0, pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to classify")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E))
    
    def _load_sample_images(self):
        """Create and load sample images"""
        self.sample_images = {}
        
        # Create sample images
        samples = {
            "Green Organic (Wet)": self._create_green_sample(),
            "Brown Organic (Wet)": self._create_brown_sample(),
            "White Paper (Dry)": self._create_white_sample(),
            "Blue Plastic (Dry)": self._create_blue_sample(),
            "Mixed Waste": self._create_mixed_sample()
        }
        
        self.sample_images = samples
        self.sample_combo['values'] = list(samples.keys())
        
        # Load first sample
        if samples:
            self.sample_var.set(list(samples.keys())[0])
            self._load_sample()
    
    def _create_green_sample(self):
        """Create green organic sample"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, 1] = 120  # Green
        img[:, :, 0] = 40   # Blue
        # Add texture
        for i in range(0, 200, 15):
            cv2.line(img, (i, 0), (i, 200), (20, 80, 20), 2)
        return img
    
    def _create_brown_sample(self):
        """Create brown organic sample"""
        img = np.full((200, 200, 3), [19, 69, 139], dtype=np.uint8)
        # Add organic texture
        for i in range(50):
            x, y = np.random.randint(0, 200, 2)
            cv2.circle(img, (x, y), np.random.randint(3, 10), (30, 50, 100), -1)
        return img
    
    def _create_white_sample(self):
        """Create white paper sample"""
        img = np.full((200, 200, 3), 220, dtype=np.uint8)
        # Add paper texture
        cv2.rectangle(img, (20, 20), (180, 180), (180, 180, 180), 3)
        cv2.line(img, (40, 60), (160, 60), (150, 150, 150), 1)
        cv2.line(img, (40, 80), (160, 80), (150, 150, 150), 1)
        cv2.line(img, (40, 100), (160, 100), (150, 150, 150), 1)
        return img
    
    def _create_blue_sample(self):
        """Create blue plastic sample"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # Blue
        img[:, :, 1] = 50   # Green
        # Add plastic texture
        cv2.rectangle(img, (30, 30), (170, 170), (150, 30, 30), 5)
        cv2.circle(img, (100, 100), 40, (100, 20, 20), 3)
        return img
    
    def _create_mixed_sample(self):
        """Create mixed waste sample"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Mix of colors
        img[:100, :100] = [50, 150, 50]  # Green
        img[:100, 100:] = [200, 200, 200]  # White
        img[100:, :100] = [200, 50, 50]  # Blue
        img[100:, 100:] = [50, 80, 160]  # Brown
        return img
    
    def _load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    self.current_image = img
                    self._display_image(img)
                    self.classify_btn.config(state="normal")
                    self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "Could not load image")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {e}")
    
    def _load_sample(self, event=None):
        """Load selected sample image"""
        sample_name = self.sample_var.get()
        if sample_name in self.sample_images:
            self.current_image = self.sample_images[sample_name]
            self._display_image(self.current_image)
            self.classify_btn.config(state="normal")
            self.status_var.set(f"Loaded sample: {sample_name}")
    
    def _display_image(self, image):
        """Display image in UI"""
        try:
            # Resize for display
            display_img = cv2.resize(image, (200, 200))
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update display
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            self.status_var.set(f"Display error: {e}")
    
    def _classify_current(self):
        """Classify current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        try:
            # Perform classification
            waste_type, confidence, processing_time = self.classifier.classify_waste(self.current_image)
            
            if waste_type:
                # Update results
                color = "#2E7D32" if waste_type == "Dry Waste" else "#8D6E63"
                self.result_label.config(text=f"Result: {waste_type}", foreground=color)
                self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
                self.time_label.config(text=f"Processing: {processing_time:.3f}s")
                
                # Save to database
                self.data_manager.save_classification(waste_type, confidence, None, processing_time)
                
                self.status_var.set(f"Classified as {waste_type} ({confidence:.1%})")
                
            else:
                messagebox.showerror("Error", "Classification failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Classification error: {e}")
    
    def _show_stats(self):
        """Show statistics window"""
        stats = self.data_manager.get_classification_stats()
        recent = self.data_manager.get_recent_classifications(10)
        
        # Create stats window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Classification Statistics")
        stats_window.geometry("400x300")
        
        # Create text widget
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add statistics
        text_widget.insert(tk.END, "Classification Statistics\\n")
        text_widget.insert(tk.END, "=" * 30 + "\\n\\n")
        
        for waste_type, data in stats.items():
            text_widget.insert(tk.END, f"{waste_type}:\\n")
            text_widget.insert(tk.END, f"  Count: {data['count']}\\n")
            text_widget.insert(tk.END, f"  Avg Confidence: {data['avg_confidence']:.1%}\\n\\n")
        
        text_widget.insert(tk.END, "Recent Classifications:\\n")
        text_widget.insert(tk.END, "-" * 25 + "\\n")
        
        for timestamp, waste_type, confidence, proc_time in recent:
            text_widget.insert(tk.END, f"{timestamp}: {waste_type} ({confidence:.1%})\\n")
        
        text_widget.config(state=tk.DISABLED)

def main():
    """Main function"""
    root = tk.Tk()
    app = TestWasteClassificationUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()