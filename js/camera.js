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
            // Detect mobile device
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
            
            // Mobile-optimized constraints
            const constraints = {
                video: {
                    width: isMobile ? { ideal: 1280, min: 480 } : { ideal: 1920, min: 640 },
                    height: isMobile ? { ideal: 720, min: 360 } : { ideal: 1080, min: 480 },
                    facingMode: 'environment', // Prefer back camera on mobile
                    frameRate: isMobile ? { ideal: 15, max: 30 } : { ideal: 30 }
                },
                audio: false
            };
            
            // iOS specific handling
            if (isIOS) {
                // iOS Safari requires user interaction before camera access
                constraints.video.width = { ideal: 1280, max: 1920 };
                constraints.video.height = { ideal: 720, max: 1080 };
            }
            
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
            
            // Try progressive fallback for mobile compatibility
            const fallbackOptions = [
                // Basic mobile constraints
                { video: { width: 640, height: 480, facingMode: 'environment' }, audio: false },
                // iOS compatible constraints
                { video: { width: 480, height: 360 }, audio: false },
                // Minimal constraints for older devices
                { video: true, audio: false }
            ];
            
            for (let i = 0; i < fallbackOptions.length; i++) {
                try {
                    console.log(`Trying fallback option ${i + 1}:`, fallbackOptions[i]);
                    this.stream = await navigator.mediaDevices.getUserMedia(fallbackOptions[i]);
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
                    console.error(`Fallback option ${i + 1} failed:`, fallbackError);
                    if (i === fallbackOptions.length - 1) {
                        this.showMobileErrorMessage(error);
                        return false;
                    }
                }
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
    
    // Mobile-specific error handling
    showMobileErrorMessage(error) {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        
        let message = 'Camera access failed. ';
        
        if (error.name === 'NotAllowedError') {
            if (isIOS) {
                message += 'On iOS: Go to Settings > Safari > Camera, and allow camera access. Then refresh this page.';
            } else if (isMobile) {
                message += 'Please allow camera permission in your browser settings and refresh the page.';
            } else {
                message += 'Please allow camera permission and refresh the page.';
            }
        } else if (error.name === 'NotFoundError') {
            message += 'No camera found. Please ensure your device has a camera.';
        } else if (error.name === 'NotSupportedError') {
            if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
                message += 'Camera requires HTTPS. Please use the HTTPS version of this site.';
            } else {
                message += 'Camera not supported in this browser. Try Chrome or Safari.';
            }
        } else {
            message += `Error: ${error.message}`;
        }
        
        // Show error in UI
        const errorDiv = document.createElement('div');
        errorDiv.className = 'mobile-error-message';
        errorDiv.innerHTML = `
            <div class="error-content">
                <h3>📱 Camera Issue</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove(); window.location.reload();">Try Again</button>
            </div>
        `;
        document.body.appendChild(errorDiv);
    }
    
    // Check if device has camera capabilities
    static async checkCameraSupport() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoInputs = devices.filter(device => device.kind === 'videoinput');
            return {
                supported: !!navigator.mediaDevices?.getUserMedia,
                hasCamera: videoInputs.length > 0,
                cameraCount: videoInputs.length,
                isSecureContext: window.isSecureContext
            };
        } catch (error) {
            return {
                supported: false,
                hasCamera: false,
                cameraCount: 0,
                isSecureContext: window.isSecureContext,
                error: error.message
            };
        }
    }
}