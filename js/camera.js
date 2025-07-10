// Camera Management Module
export class CameraManager {
    constructor() {
        this.video = null;
        this.stream = null;
        this.canvas = null;
        this.ctx = null;
        this.active = false;
    }
    
    init() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('detectionCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set up video event listeners
        this.video.addEventListener('loadedmetadata', () => {
            this.updateCanvasSize();
        });
        
        this.video.addEventListener('resize', () => {
            this.updateCanvasSize();
        });
    }
    
    async start() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 1920, min: 640 },
                    height: { ideal: 1080, min: 480 },
                    facingMode: 'environment' // Prefer back camera on mobile
                },
                audio: false
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.active = true;
                    this.updateCanvasSize();
                    resolve(true);
                };
            });
            
        } catch (error) {
            console.error('Camera access error:', error);
            
            // Try with reduced constraints if initial attempt fails
            try {
                const fallbackConstraints = {
                    video: { width: 640, height: 480 },
                    audio: false
                };
                
                this.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                this.video.srcObject = this.stream;
                
                return new Promise((resolve) => {
                    this.video.onloadedmetadata = () => {
                        this.video.play();
                        this.active = true;
                        this.updateCanvasSize();
                        resolve(true);
                    };
                });
                
            } catch (fallbackError) {
                console.error('Fallback camera access failed:', fallbackError);
                return false;
            }
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        this.active = false;
        
        // Clear canvas
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
    
    getCurrentFrame() {
        if (!this.active || !this.video || this.video.readyState !== 4) {
            return null;
        }
        
        // Create a temporary canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        
        tempCtx.drawImage(this.video, 0, 0);
        
        return tempCanvas;
    }
    
    getCurrentFrameImageData() {
        const frame = this.getCurrentFrame();
        if (!frame) return null;
        
        const ctx = frame.getContext('2d');
        return ctx.getImageData(0, 0, frame.width, frame.height);
    }
    
    updateCanvasSize() {
        if (!this.video || !this.canvas) return;
        
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Maintain aspect ratio
        const rect = this.video.getBoundingClientRect();
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }
    
    isActive() {
        return this.active && this.video && this.video.readyState === 4;
    }
    
    getVideoElement() {
        return this.video;
    }
    
    getCanvasElement() {
        return this.canvas;
    }
    
    getContext() {
        return this.ctx;
    }
    
    // Get available camera devices
    async getAvailableCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error enumerating devices:', error);
            return [];
        }
    }
    
    // Switch to a specific camera
    async switchCamera(deviceId) {
        if (this.active) {
            this.stop();
        }
        
        try {
            const constraints = {
                video: {
                    deviceId: { exact: deviceId },
                    width: { ideal: 1920, min: 640 },
                    height: { ideal: 1080, min: 480 }
                },
                audio: false
            };
            
            return await this.start();
            
        } catch (error) {
            console.error('Camera switch error:', error);
            return false;
        }
    }
    
    // Take a screenshot
    takeScreenshot() {
        const frame = this.getCurrentFrame();
        if (!frame) return null;
        
        return frame.toDataURL('image/jpeg', 0.9);
    }
    
    // Get camera capabilities
    getCapabilities() {
        if (!this.stream) return null;
        
        const videoTrack = this.stream.getVideoTracks()[0];
        if (!videoTrack) return null;
        
        return videoTrack.getCapabilities();
    }
    
    // Get current camera settings
    getSettings() {
        if (!this.stream) return null;
        
        const videoTrack = this.stream.getVideoTracks()[0];
        if (!videoTrack) return null;
        
        return videoTrack.getSettings();
    }
}