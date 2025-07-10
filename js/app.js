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
        
        // Load AI models
        await this.loadAIModels();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize components
        this.camera.init();
        this.arduino.init();
        
        // Setup Arduino status updates
        this.arduino.onStatusChange = (status) => {
            this.updateArduinoStatus(status);
        };
        
        // Show main interface
        this.showMainInterface();
        
        console.log('✅ App initialized successfully!');
    }
    
    async loadAIModels() {
        const loadingDetails = document.getElementById('loadingDetails');
        
        try {
            loadingDetails.textContent = 'Loading MobileNet (Image Classification)...';
            await this.aiDetector.loadMobileNet();
            
            loadingDetails.textContent = 'Loading COCO-SSD (Object Detection)...';
            await this.aiDetector.loadCocoSSD();
            
            loadingDetails.textContent = 'AI models ready! 🚀';
            
        } catch (error) {
            console.error('Failed to load AI models:', error);
            loadingDetails.textContent = 'Error loading AI models. Please refresh the page.';
            this.updateStatus('❌ Failed to load AI models', 'error');
            throw error;
        }
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
document.addEventListener('DOMContentLoaded', () => {
    new WasteDetectionApp();
});