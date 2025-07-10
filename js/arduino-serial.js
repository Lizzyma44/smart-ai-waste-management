// Arduino Serial Communication Module using Web Serial API
export class ArduinoSerial {
    constructor() {
        this.port = null;
        this.reader = null;
        this.writer = null;
        this.connected = false;
        this.connecting = false;
        this.onStatusChange = null;
        this.commandQueue = [];
        this.lastCommand = null;
        this.lastCommandTime = null;
    }
    
    init() {
        // Check if Web Serial API is supported
        if (!('serial' in navigator)) {
            console.warn('Web Serial API not supported in this browser');
            this.updateStatus('disconnected');
            return false;
        }
        
        console.log('✅ Web Serial API supported');
        this.updateStatus('disconnected');
        return true;
    }
    
    async connect(baudRate = 9600) {
        if (!('serial' in navigator)) {
            throw new Error('Web Serial API not supported');
        }
        
        if (this.connected) {
            console.log('Already connected to Arduino');
            return true;
        }
        
        this.connecting = true;
        this.updateStatus('connecting');
        
        try {
            // Request a port
            this.port = await navigator.serial.requestPort();
            
            // Open the port
            await this.port.open({ 
                baudRate: baudRate,
                dataBits: 8,
                stopBits: 1,
                parity: 'none',
                flowControl: 'none'
            });
            
            // Set up streams
            this.reader = this.port.readable.getReader();
            this.writer = this.port.writable.getWriter();
            
            this.connected = true;
            this.connecting = false;
            this.updateStatus('connected');
            
            // Start listening for incoming data
            this.startReading();
            
            // Send initial status request
            setTimeout(() => {
                this.sendCommand('s'); // Status command
            }, 1000);
            
            console.log('✅ Arduino connected successfully');
            return true;
            
        } catch (error) {
            this.connecting = false;
            this.updateStatus('disconnected');
            console.error('Arduino connection failed:', error);
            
            if (error.name === 'NotFoundError') {
                throw new Error('No Arduino found. Please connect your Arduino via USB.');
            } else if (error.name === 'NetworkError') {
                throw new Error('Failed to open serial port. Arduino may be in use by another application.');
            } else {
                throw new Error(`Connection failed: ${error.message}`);
            }
        }
    }
    
    async disconnect() {
        if (!this.connected) {
            return;
        }
        
        try {
            // Close reader and writer
            if (this.reader) {
                await this.reader.cancel();
                await this.reader.releaseLock();
                this.reader = null;
            }
            
            if (this.writer) {
                await this.writer.releaseLock();
                this.writer = null;
            }
            
            // Close port
            if (this.port) {
                await this.port.close();
                this.port = null;
            }
            
            this.connected = false;
            this.updateStatus('disconnected');
            
            console.log('Arduino disconnected');
            
        } catch (error) {
            console.error('Error during disconnect:', error);
        }
    }
    
    async sendCommand(command) {
        if (!this.connected || !this.writer) {
            console.warn('Cannot send command: Arduino not connected');
            return false;
        }
        
        try {
            // Convert command to Uint8Array
            const data = new TextEncoder().encode(command);
            
            // Send command
            await this.writer.write(data);
            
            // Log command
            this.lastCommand = command;
            this.lastCommandTime = new Date();
            
            console.log(`📤 Sent command: ${command}`);
            return true;
            
        } catch (error) {
            console.error('Error sending command:', error);
            return false;
        }
    }
    
    async startReading() {
        if (!this.reader) return;
        
        try {
            while (this.connected) {
                const { value, done } = await this.reader.read();
                
                if (done) {
                    console.log('Arduino stream closed');
                    break;
                }
                
                // Decode incoming data
                const text = new TextDecoder().decode(value);
                this.handleIncomingData(text);
            }
        } catch (error) {
            if (error.name !== 'NetworkError') {
                console.error('Error reading from Arduino:', error);
            }
        }
    }
    
    handleIncomingData(data) {
        console.log(`📥 Received: ${data.trim()}`);
        
        // Parse Arduino responses
        const lines = data.split('\n');
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed) {
                this.parseArduinoResponse(trimmed);
            }
        }
    }
    
    parseArduinoResponse(response) {
        // Handle different Arduino responses
        if (response.startsWith('STATUS:')) {
            const status = response.replace('STATUS:', '');
            console.log(`Arduino status: ${status}`);
        } else if (response.includes('wet waste')) {
            console.log('✅ Arduino confirmed wet waste processing');
        } else if (response.includes('dry waste')) {
            console.log('✅ Arduino confirmed dry waste processing');
        } else if (response.includes('ready') || response.includes('Ready')) {
            console.log('✅ Arduino ready for commands');
        } else {
            console.log(`Arduino: ${response}`);
        }
    }
    
    async getAvailablePorts() {
        if (!('serial' in navigator)) {
            return [];
        }
        
        try {
            const ports = await navigator.serial.getPorts();
            return ports;
        } catch (error) {
            console.error('Error getting ports:', error);
            return [];
        }
    }
    
    isConnected() {
        return this.connected;
    }
    
    isConnecting() {
        return this.connecting;
    }
    
    getLastCommand() {
        return {
            command: this.lastCommand,
            timestamp: this.lastCommandTime
        };
    }
    
    updateStatus(status) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }
    
    // Queue commands to prevent overwhelming Arduino
    async queueCommand(command) {
        this.commandQueue.push(command);
        
        if (this.commandQueue.length === 1) {
            await this.processCommandQueue();
        }
    }
    
    async processCommandQueue() {
        while (this.commandQueue.length > 0) {
            const command = this.commandQueue.shift();
            await this.sendCommand(command);
            
            // Wait a bit between commands
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    // Send waste detection result to Arduino
    async sendWasteDetection(wasteType, confidence) {
        let command;
        
        if (wasteType === 'WET WASTE') {
            command = 'w';
        } else if (wasteType === 'DRY WASTE') {
            command = 'd';
        } else {
            console.warn('Unknown waste type:', wasteType);
            return false;
        }
        
        return await this.sendCommand(command);
    }
    
    // Test Arduino connection
    async testConnection() {
        if (!this.connected) {
            return false;
        }
        
        try {
            // Send status command and wait for response
            await this.sendCommand('s');
            
            // Wait a moment for response
            await new Promise(resolve => setTimeout(resolve, 500));
            
            return true;
        } catch (error) {
            console.error('Connection test failed:', error);
            return false;
        }
    }
    
    // Get connection info
    getConnectionInfo() {
        if (!this.port) {
            return null;
        }
        
        return {
            connected: this.connected,
            connecting: this.connecting,
            portInfo: this.port.getInfo(),
            lastCommand: this.lastCommand,
            lastCommandTime: this.lastCommandTime
        };
    }
    
    // Handle connection errors
    handleConnectionError(error) {
        console.error('Arduino connection error:', error);
        
        if (this.connected) {
            this.disconnect();
        }
        
        this.updateStatus('disconnected');
    }
    
    // Auto-reconnect functionality
    async autoReconnect(maxAttempts = 3) {
        if (this.connected || this.connecting) {
            return false;
        }
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            console.log(`Reconnection attempt ${attempt}/${maxAttempts}`);
            
            try {
                await this.connect();
                console.log('✅ Reconnected successfully');
                return true;
            } catch (error) {
                console.error(`Reconnection attempt ${attempt} failed:`, error);
                
                if (attempt < maxAttempts) {
                    // Wait before next attempt
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }
        
        console.error('All reconnection attempts failed');
        return false;
    }
}