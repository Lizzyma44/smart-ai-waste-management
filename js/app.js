// Smart Waste Detection - Main Application
import { CameraManager } from './camera.js';
import { AIDetector } from './ai-detection.js';
import { ArduinoSerial } from './arduino-serial.js';

class WasteDetectionApp {
    constructor() {
        this.camera = new CameraManager();
        this.aiDetector = new AIDetector();
        this.arduino = new ArduinoSerial();
        
        this.isDetecting = false;
        this.autoDetectInterval = null;
        this.totalDetections = 0;
        this.totalConfidence = 0;
        
        this.init();
    }
    
    async init() {
        console.log('🤖 Initializing Smart Waste Detection App...');
        
        try {
            // Check mobile camera support early
            if (this.isMobile) {
                const cameraSupport = await CameraManager.checkCameraSupport();
                if (!cameraSupport.supported || !cameraSupport.isSecureContext) {
                    this.showMobileError(cameraSupport);
                    return;
                }
            }
            
            // Load AI models with error handling
            try {
                await this.loadAIModels();
                this.aiAvailable = true;
            } catch (error) {
                console.warn('AI models failed to load, continuing with limited functionality');
                this.aiAvailable = false;
                this.setupLimitedMode();
            }
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize components
            this.camera.init();
            this.arduino.init();
            
            // Setup Arduino status updates
            this.arduino.onStatusChange = (status) => {
                this.updateArduinoStatus(status);
            };
            
            // Mobile-specific setup
            if (this.isMobile) {
                this.setupMobileFeatures();
            }
            
            // Show main interface
            this.showMainInterface();
            
            console.log('✅ App initialized successfully!');
            
        } catch (error) {
            console.error('❌ App initialization failed:', error);
            // Don't throw - let the error display handle it
            // The loading screen will show the error message
        }
    }
    
    setupLimitedMode() {
        // Show limited mode interface
        const loadingScreen = document.getElementById('loadingScreen');
        loadingScreen.innerHTML = `
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">⚠️ Limited Mode</div>
                <div class="loading-details">
                    <div class="error-details">
                        <p class="error-message">AI models couldn't load, but you can still use the camera!</p>
                        <p class="error-suggestion">Camera and Arduino features are available. Manual waste classification only.</p>
                        <button class="retry-button" onclick="this.parentElement.parentElement.parentElement.parentElement.style.display='none'; document.getElementById('mainInterface').style.display='block';">
                            📱 Continue with Camera Only
                        </button>
                        <button class="retry-button" onclick="window.location.reload()">
                            🔄 Try Loading AI Again
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Disable AI-dependent features
        this.limitedMode = true;
    }
    
    async loadAIModels() {
        const loadingDetails = document.getElementById('loadingDetails');
        const loadingText = document.querySelector('.loading-text');
        
        try {
            // Check TensorFlow.js support first
            loadingDetails.textContent = 'Checking AI framework compatibility...';
            
            // Wait for TensorFlow.js to be available
            let tfCheckAttempts = 0;
            while (typeof tf === 'undefined' && tfCheckAttempts < 10) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                tfCheckAttempts++;
            }
            
            const tfSupport = await AIDetector.checkTensorFlowSupport();
            
            if (!tfSupport.supported) {
                throw new Error(`AI framework not supported: ${tfSupport.error}`);
            }
            
            loadingDetails.textContent = `Backend: ${tfSupport.backend}, WebGL: ${tfSupport.webgl ? 'Yes' : 'No'}`;
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Load MobileNet with progress
            loadingText.textContent = 'Loading Image Classification AI...';
            loadingDetails.textContent = this.isMobile ? 
                'Loading optimized model for mobile...' : 
                'Loading MobileNet (Image Classification)...';
            
            await this.loadModelWithProgress('mobilenet', () => this.aiDetector.loadMobileNet());
            
            // Load COCO-SSD with progress
            loadingText.textContent = 'Loading Object Detection AI...';
            loadingDetails.textContent = this.isMobile ? 
                'Loading object detection for mobile...' : 
                'Loading COCO-SSD (Object Detection)...';
            
            await this.loadModelWithProgress('cocossd', () => this.aiDetector.loadCocoSSD());
            
            // Success
            loadingText.textContent = 'AI Models Ready!';
            loadingDetails.textContent = '✅ All AI models loaded successfully! Ready to detect waste.';
            
        } catch (error) {
            console.error('Failed to load AI models:', error);
            this.handleAILoadingError(error, loadingText, loadingDetails);
            throw error;
        }
    }
    
    async loadModelWithProgress(modelName, loadFunction) {
        const startTime = Date.now();
        let progressInterval;
        
        try {
            // Start progress animation
            progressInterval = this.startLoadingProgress(modelName);
            
            // Load the model
            await loadFunction();
            
            const loadTime = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(`✅ ${modelName} loaded in ${loadTime}s`);
            
        } finally {
            if (progressInterval) {
                clearInterval(progressInterval);
            }
        }
    }
    
    startLoadingProgress(modelName) {
        const loadingDetails = document.getElementById('loadingDetails');
        let dots = 0;
        
        return setInterval(() => {
            dots = (dots + 1) % 4;
            const dotString = '.'.repeat(dots);
            const baseMessage = this.isMobile ? 
                `Loading ${modelName} for mobile` : 
                `Loading ${modelName} model`;
            loadingDetails.textContent = `${baseMessage}${dotString}`;
        }, 500);
    }
    
    handleAILoadingError(error, loadingText, loadingDetails) {
        loadingText.textContent = '❌ AI Loading Failed';
        
        let errorMessage = 'Unknown error occurred';
        let suggestion = 'Please refresh the page to try again.';
        
        if (error.message.includes('timeout')) {
            errorMessage = this.isMobile ? 
                'Mobile connection too slow for AI models' : 
                'Network timeout loading AI models';
            suggestion = 'Try connecting to a faster internet connection and refresh.';
        } else if (error.message.includes('TensorFlow.js not loaded')) {
            errorMessage = 'AI framework failed to load';
            suggestion = 'Check your internet connection and refresh the page.';
        } else if (error.message.includes('WebGL')) {
            errorMessage = 'Your device does not support WebGL acceleration';
            suggestion = this.isMobile ? 
                'Try using Chrome browser or a newer device.' : 
                'Try updating your browser or graphics drivers.';
        } else if (error.message.includes('MobileNet')) {
            errorMessage = 'Image classification AI failed to load';
            suggestion = 'The AI service may be temporarily unavailable. Try again later.';
        } else if (error.message.includes('COCO-SSD')) {
            errorMessage = 'Object detection AI failed to load';
            suggestion = 'The AI service may be temporarily unavailable. Try again later.';
        }
        
        loadingDetails.innerHTML = `
            <div class="error-details">
                <p class="error-message">⚠️ ${errorMessage}</p>
                <p class="error-suggestion">${suggestion}</p>
                <button class="retry-button" onclick="window.location.reload()">
                    🔄 Refresh Page
                </button>
                ${this.isMobile ? 
                    '<p class="mobile-note"><small>Note: AI models require stable internet on mobile devices</small></p>' : 
                    ''
                }
            </div>
        `;
        
        this.updateStatus('❌ Failed to load AI models', 'error');
    }
    
    showMainInterface() {
        document.getElementById('loadingScreen').style.display = 'none';
        document.getElementById('mainInterface').style.display = 'block';
        document.getElementById('mainInterface').classList.add('fade-in');
        this.updateStatus('✨ Ready! Click "Start Camera" to begin detection.', 'success');
    }
    
    setupEventListeners() {
        // Camera controls
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('detectBtn').addEventListener('click', () => this.detectWaste());
        
        // Auto detection toggle
        document.getElementById('autoToggle').addEventListener('change', (e) => {
            this.toggleAutoDetect(e.target.checked);
        });
        
        // Arduino controls
        document.getElementById('connectArduinoBtn').addEventListener('click', () => this.connectArduino());
        
        // Help panel
        document.getElementById('helpBtn').addEventListener('click', () => this.toggleHelp());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.target.matches('input, textarea, button, select')) {
                e.preventDefault();
                this.detectWaste();
            } else if (e.code === 'KeyC' && e.ctrlKey) {
                e.preventDefault();
                this.connectArduino();
            }
        });
        
        // Close help when clicking outside
        document.addEventListener('click', (e) => {
            const helpContent = document.getElementById('helpContent');
            const helpBtn = document.getElementById('helpBtn');
            if (helpContent.style.display === 'block' && 
                !helpContent.contains(e.target) && 
                !helpBtn.contains(e.target)) {
                helpContent.style.display = 'none';
            }
        });
    }
    
    async startCamera() {
        try {
            this.updateStatus('🎥 Starting camera...', 'info');
            
            const success = await this.camera.start();
            if (success) {
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('detectBtn').disabled = false;
                
                document.getElementById('wasteCategoryLarge').textContent = 'CAMERA READY';
                this.updateStatus('✅ Camera started! Point at waste items for detection.', 'success');
            } else {
                throw new Error('Camera initialization failed');
            }
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('❌ Camera access denied. Please allow camera permissions.', 'error');
        }
    }
    
    stopCamera() {
        this.camera.stop();
        this.stopAutoDetect();
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('detectBtn').disabled = true;
        
        document.getElementById('wasteCategoryLarge').textContent = 'READY';
        document.getElementById('wasteCategoryLarge').className = 'waste-category-large unknown';
        document.getElementById('confidenceValue').textContent = '--';
        document.getElementById('processingValue').textContent = '--';
        
        this.updateStatus('📱 Camera stopped', 'info');
    }
    
    async detectWaste() {
        if (this.isDetecting || !this.camera.isActive()) {
            return;
        }
        
        this.isDetecting = true;
        const startTime = performance.now();
        
        try {
            // Check if AI is available
            if (!this.aiAvailable || this.limitedMode) {
                // Manual mode - let user classify
                document.getElementById('wasteCategoryLarge').textContent = 'CAMERA ONLY';
                this.showManualClassification();
                return;
            }
            
            // Update UI
            document.getElementById('wasteCategoryLarge').textContent = 'ANALYZING...';
            document.getElementById('infoPanel').classList.add('detection-animation');
            
            // Capture frame
            const frame = this.camera.getCurrentFrame();
            if (!frame) {
                throw new Error('No camera frame available');
            }
            
            // Run AI detection
            const result = await this.aiDetector.classifyWaste(frame);
            const processingTime = (performance.now() - startTime) / 1000;
            
            // Update results
            this.updateResults({
                ...result,
                processingTime
            });
            
            // Send Arduino command
            if (this.arduino.isConnected()) {
                const command = result.wasteCategory === 'WET WASTE' ? 'w' : 'd';
                await this.arduino.sendCommand(command);
                this.logArduinoCommand(command, result.wasteCategory);
            }
            
            // Update statistics
            this.updateStatistics(result);
            
            this.updateStatus(`🎯 Detected: ${result.wasteCategory} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
            
        } catch (error) {
            console.error('Detection error:', error);
            this.updateStatus('❌ Detection failed', 'error');
            document.getElementById('wasteCategoryLarge').textContent = 'ERROR';
            document.getElementById('wasteCategoryLarge').className = 'waste-category-large unknown';
        } finally {
            this.isDetecting = false;
            document.getElementById('infoPanel').classList.remove('detection-animation');
        }
    }
    
    toggleAutoDetect(enabled) {
        if (enabled && this.camera.isActive()) {
            this.startAutoDetect();
        } else {
            this.stopAutoDetect();
        }
    }
    
    showManualClassification() {
        // Show manual classification options
        const wasteCategoryLarge = document.getElementById('wasteCategoryLarge');
        wasteCategoryLarge.innerHTML = `
            <div class="manual-classification">
                <p style="font-size: 0.8em; margin-bottom: 15px;">Manual Classification</p>
                <button class="manual-btn wet-btn" onclick="app.classifyManually('wet')">
                    🥬 WET WASTE
                </button>
                <button class="manual-btn dry-btn" onclick="app.classifyManually('dry')">
                    📦 DRY WASTE
                </button>
            </div>
        `;
        
        this.isDetecting = false;
        this.updateStatus('📱 Camera-only mode: Manually classify the waste', 'info');
    }
    
    classifyManually(wasteType) {
        const result = {
            wasteCategory: wasteType.toUpperCase() + ' WASTE',
            object: 'manually classified',
            confidence: 1.0,
            source: 'manual',
            processingTime: 0
        };
        
        this.updateResults(result);
        
        // Send Arduino command
        if (this.arduino.isConnected()) {
            const command = wasteType === 'wet' ? 'w' : 'd';
            this.arduino.sendCommand(command);
        }
        
        // Show toast feedback
        if (this.isMobile) {
            this.showToast(`✅ Classified as ${wasteType} waste`);
        }
        
        // Reset to camera view after 2 seconds
        setTimeout(() => {
            document.getElementById('wasteCategoryLarge').textContent = 'CAMERA ONLY';
        }, 2000);
    }
    
    startAutoDetect() {
        if (this.autoDetectInterval) return;
        
        this.autoDetectInterval = setInterval(() => {
            if (this.camera.isActive() && !this.isDetecting) {
                this.detectWaste();
            }
        }, 3000);
        
        document.getElementById('autoIndicator').style.display = 'block';
        this.updateStatus('🔄 Auto-detection enabled - analyzing every 3 seconds', 'success');
    }
    
    stopAutoDetect() {
        if (this.autoDetectInterval) {
            clearInterval(this.autoDetectInterval);
            this.autoDetectInterval = null;
        }
        
        document.getElementById('autoIndicator').style.display = 'none';
        document.getElementById('autoToggle').checked = false;
    }
    
    async connectArduino() {
        const connectBtn = document.getElementById('connectArduinoBtn');
        const baudRate = document.getElementById('baudRateSelect').value;
        
        if (this.arduino.isConnected()) {
            // Disconnect
            this.arduino.disconnect();
            connectBtn.textContent = '🔌 Connect Arduino';
        } else {
            // Connect
            connectBtn.disabled = true;
            connectBtn.textContent = '🔄 Connecting...';
            
            try {
                const success = await this.arduino.connect(parseInt(baudRate));
                if (success) {
                    connectBtn.textContent = '🔌 Disconnect';
                    this.updateStatus('✅ Arduino connected successfully!', 'success');
                } else {
                    throw new Error('Connection failed');
                }
            } catch (error) {
                console.error('Arduino connection error:', error);
                this.updateStatus('❌ Arduino connection failed', 'error');
                connectBtn.textContent = '🔌 Connect Arduino';
            } finally {
                connectBtn.disabled = false;
            }
        }
    }
    
    updateResults(result) {
        const wasteCategoryElement = document.getElementById('wasteCategoryLarge');
        wasteCategoryElement.textContent = result.wasteCategory;
        
        if (result.wasteCategory === 'DRY WASTE') {
            wasteCategoryElement.className = 'waste-category-large dry';
        } else if (result.wasteCategory === 'WET WASTE') {
            wasteCategoryElement.className = 'waste-category-large wet';
        } else {
            wasteCategoryElement.className = 'waste-category-large unknown';
        }
        
        document.getElementById('confidenceValue').textContent = (result.confidence * 100).toFixed(0) + '%';
        document.getElementById('processingValue').textContent = result.processingTime.toFixed(2);
    }
    
    updateStatistics(result) {
        this.totalDetections++;
        this.totalConfidence += result.confidence;
        
        document.getElementById('totalCount').textContent = this.totalDetections;
        
        const avgConfidence = (this.totalConfidence / this.totalDetections) * 100;
        document.getElementById('accuracyValue').textContent = avgConfidence.toFixed(0) + '%';
    }
    
    updateArduinoStatus(status) {
        const statusElement = document.getElementById('arduinoStatus');
        const connectBtn = document.getElementById('connectArduinoBtn');
        
        switch (status) {
            case 'connected':
                statusElement.textContent = '🔌 Arduino Connected';
                statusElement.className = 'arduino-status connected';
                connectBtn.textContent = '🔌 Disconnect';
                break;
            case 'connecting':
                statusElement.textContent = '🔄 Arduino Connecting...';
                statusElement.className = 'arduino-status connecting';
                break;
            case 'disconnected':
            default:
                statusElement.textContent = '🔌 Arduino Disconnected';
                statusElement.className = 'arduino-status';
                connectBtn.textContent = '🔌 Connect Arduino';
                break;
        }
    }
    
    logArduinoCommand(command, wasteType) {
        const commandLog = document.getElementById('commandLog');
        const timestamp = new Date().toLocaleTimeString();
        const message = `${timestamp}: Sent '${command}' (${wasteType})`;
        
        // Add new message
        if (commandLog.textContent === 'No commands sent yet') {
            commandLog.textContent = message;
        } else {
            commandLog.textContent = message + '\n' + commandLog.textContent;
        }
        
        // Keep only last 5 messages
        const lines = commandLog.textContent.split('\n');
        if (lines.length > 5) {
            commandLog.textContent = lines.slice(0, 5).join('\n');
        }
    }
    
    updateStatus(message, type) {
        const statusIndicator = document.getElementById('statusIndicator');
        statusIndicator.textContent = message;
        statusIndicator.className = `status-indicator ${type}`;
    }
    
    toggleHelp() {
        const helpContent = document.getElementById('helpContent');
        helpContent.style.display = helpContent.style.display === 'none' ? 'block' : 'none';
    }
}

// Initialize app when DOM is loaded
let app; // Make app globally accessible for manual classification
document.addEventListener('DOMContentLoaded', () => {
    app = new WasteDetectionApp();
});