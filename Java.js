 // Global variables
        let video = null;
        let detectionCanvas = null;
        let detectionCtx = null;
        let stream = null;
        let mobileNetModel = null;
        let cocoSsdModel = null;
        let isDetecting = false;
        let autoDetectInterval = null;
        let totalDetections = 0;
        let totalConfidence = 0;

        // Serial communication variables
        let serialPort = null;
        let writer = null;
        let isArduinoConnected = false;

        // Waste classification mapping
        const wasteClassification = {
            // Wet Waste
            'banana': 'wet', 'apple': 'wet', 'orange': 'wet', 'broccoli': 'wet', 'carrot': 'wet',
            'hot dog': 'wet', 'pizza': 'wet', 'donut': 'wet', 'cake': 'wet', 'avocado': 'wet',
            'lemon': 'wet', 'sandwich': 'wet', 'food': 'wet', 'fruit': 'wet', 'vegetable': 'wet',
            'plant': 'wet', 'flower': 'wet', 'bread': 'wet', 'egg': 'wet', 'mushroom': 'wet',
            'salad': 'wet', 'meat': 'wet', 'cheese': 'wet', 'seafood': 'wet',

            // Paper Waste
            'book': 'paper', 'paper': 'paper', 'notebook': 'paper', 'newspaper': 'paper',

            // Plastic Waste
            'bottle': 'plastic', 'plastic': 'plastic', 'plastic bag': 'plastic',

            // Glass Waste
            'wine glass': 'glass', 'cup': 'glass', 'glass': 'glass', 'bowl': 'glass',

            // Electronic Waste
            'cell phone': 'electronic', 'laptop': 'electronic', 'mouse': 'electronic', 'keyboard': 'electronic', 'mouse': 'electronic',

            // Can Waste
            'can': 'can', 'aluminum': 'can', 'metal': 'can', 'soda can': 'can',

            // Unknown / Other
            'car': 'unknown', 'bicycle': 'unknown', 'bus': 'unknown', 'train': 'unknown',
            'truck': 'unknown', 'boat': 'unknown', 'traffic light': 'unknown', 'fire hydrant': 'unknown',
            'stop sign': 'unknown', 'parking meter': 'unknown', 'bench': 'unknown', 'bird': 'unknown',
            'cat': 'unknown', 'dog': 'unknown', 'horse': 'unknown', 'sheep': 'unknown', 'cow': 'unknown',
            'elephant': 'unknown', 'bear': 'unknown', 'zebra': 'unknown', 'giraffe': 'unknown'
        };


        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', async function() {
            await loadAIModels();
            initializeElements();
            setupEventListeners();
            checkSerialSupport();
        });

        async function loadAIModels() {
            try {
                document.getElementById('loadingDetails').textContent = 'Loading MobileNet (Image Classification)...';
                mobileNetModel = await mobilenet.load();
               
                document.getElementById('loadingDetails').textContent = 'Loading COCO-SSD (Object Detection)...';
                cocoSsdModel = await cocoSsd.load();
               
                document.getElementById('loadingScreen').style.display = 'none';
                document.getElementById('mainInterface').style.display = 'block';
                document.getElementById('mainInterface').classList.add('fade-in');
               
                updateStatus('✨ AI Models loaded! Connect Arduino and start camera.', 'success');
               
            } catch (error) {
                console.error('Error loading AI models:', error);
                updateStatus('❌ Failed to load AI models. Please refresh the page.', 'error');
            }
        }

        function initializeElements() {
            video = document.getElementById('video');
            detectionCanvas = document.getElementById('detectionCanvas');
            detectionCtx = detectionCanvas.getContext('2d');
        }

        function setupEventListeners() {
            document.getElementById('connectArduino').addEventListener('click', connectToArduino);
            document.getElementById('startBtn').addEventListener('click', startCamera);
            document.getElementById('stopBtn').addEventListener('click', stopCamera);
            document.getElementById('detectBtn').addEventListener('click', detectNow);
            document.getElementById('autoToggle').addEventListener('change', toggleAutoDetect);
           
            document.addEventListener('keydown', function(e) {
                if (e.code === 'Space' && !e.target.matches('input, textarea, button')) {
                    e.preventDefault();
                    detectNow();
                }
            });
        }

        function checkSerialSupport() {
            if ('serial' in navigator) {
                logSerial('✅ Web Serial API supported');
                updateArduinoStatus('Arduino Ready to Connect', 'connecting');
            } else {
                logSerial('❌ Web Serial API not supported');
                updateArduinoStatus('Serial API Not Supported', 'disconnected');
                document.getElementById('connectArduino').disabled = true;
            }
        }

        async function connectToArduino() {
            try {
                updateArduinoStatus('Connecting...', 'connecting');
                logSerial('> Requesting serial port...');
               
                serialPort = await navigator.serial.requestPort();
               
                await serialPort.open({ baudRate: 9600 });
               
                writer = serialPort.writable.getWriter();
               
                isArduinoConnected = true;
                updateArduinoStatus('Arduino Connected', 'connected');
                logSerial('✅ Arduino connected successfully');
                logSerial('> Ready to send commands');
               
                document.getElementById('connectArduino').textContent = '🔗 Connected';
                document.getElementById('connectArduino').disabled = true;
               
            } catch (error) {
                console.error('Arduino connection error:', error);
                updateArduinoStatus('Connection Failed', 'disconnected');
                logSerial('❌ Connection failed: ' + error.message);
            }
        }
       
        async function sendToArduino(command) {
            if (!isArduinoConnected) {
                logSerial('⚠️ Not connected to Arduino. Command not sent.');
                return;
            }
            try {
                const encoder = new TextEncoder();
                await writer.write(encoder.encode(command));
                logSerial(`➡️ Command sent: '${command}'`);
            } catch (error) {
                console.error('Error sending data to Arduino:', error);
                logSerial('❌ Failed to send command: ' + error.message);
            }
        }

        function logSerial(message) {
            const serialLog = document.getElementById('serialLog');
            serialLog.innerHTML += message + '<br>';
            serialLog.scrollTop = serialLog.scrollHeight;
        }

        function updateArduinoStatus(message, status) {
            const statusElement = document.getElementById('arduinoStatus');
            statusElement.textContent = `🔌 ${message}`;
            statusElement.className = `arduino-status ${status}`;
        }

        async function startCamera() {
            try {
                updateStatus('🎥 Starting camera...', 'info');
               
                const constraints = {
                    video: {
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                        facingMode: 'environment'
                    }
                };
               
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
               
                video.onloadedmetadata = () => {
                    video.play();
                   
                    detectionCanvas.width = video.videoWidth;
                    detectionCanvas.height = video.videoHeight;
                    detectionCanvas.style.width = '100%';
                    detectionCanvas.style.height = '100%';
                   
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('detectBtn').disabled = false;
                   
                    document.getElementById('wasteCategoryLarge').textContent = 'CAMERA READY';
                    updateStatus('✅ Camera started! Show items for detection.', 'success');
                   
                    logSerial('📹 Camera started - ready for detection');
                };
               
            } catch (error) {
                console.error('Camera error:', error);
                updateStatus('❌ Camera access denied. Please allow permissions.', 'error');
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
           
            if (autoDetectInterval) {
                clearInterval(autoDetectInterval);
                autoDetectInterval = null;
                document.getElementById('autoToggle').checked = false;
                document.getElementById('autoIndicator').style.display = 'none';
            }
           
            detectionCtx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
           
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('detectBtn').disabled = true;
           
            document.getElementById('wasteCategoryLarge').textContent = 'READY';
            document.getElementById('wasteCategoryLarge').className = 'waste-category-large unknown';
            document.getElementById('confidenceValue').textContent = '--';
            document.getElementById('processingValue').textContent = '--';
           
            updateStatus('📱 Camera stopped', 'info');
            logSerial('📹 Camera stopped');
        }

        function detectNow() {
            if (stream && !isDetecting) {
                detectCurrentFrame();
            }
        }

        function toggleAutoDetect() {
            const isChecked = document.getElementById('autoToggle').checked;
           
            if (isChecked && stream) {
                startAutoDetection();
            } else {
                stopAutoDetection();
            }
        }

        function startAutoDetection() {
            if (autoDetectInterval) return;
           
            autoDetectInterval = setInterval(() => {
                if (stream && !isDetecting) {
                    detectCurrentFrame();
                }
            }, 3000);
           
            document.getElementById('autoIndicator').style.display = 'block';
            updateStatus('🔄 Auto-detection enabled - analyzing every 3 seconds', 'success');
            logSerial('🔄 Auto-detection enabled');
        }

        function stopAutoDetection() {
            if (autoDetectInterval) {
                clearInterval(autoDetectInterval);
                autoDetectInterval = null;
            }
           
            document.getElementById('autoIndicator').style.display = 'none';
            updateStatus('⏸️ Auto-detection disabled', 'info');
            logSerial('⏸️ Auto-detection disabled');
        }

        async function detectCurrentFrame() {
            if (!video || !stream || isDetecting || !mobileNetModel || !cocoSsdModel) return;
           
            isDetecting = true;
            const startTime = performance.now();
           
            try {
                document.getElementById('wasteCategoryLarge').textContent = 'ANALYZING...';
                document.getElementById('infoPanel').classList.add('detection-animation');
               
                const classifications = await mobileNetModel.classify(video);
                const detections = await cocoSsdModel.detect(video);

                const processingTime = (performance.now() - startTime) / 1000;
               
                const result = processAIResults(classifications, detections, processingTime);
               
                updateResults(result);
                updateStatistics(result);
               
                await sendWasteCommand(result.wasteCategory);
               
                updateStatus(`🎯 Detected: ${result.wasteCategory} → Arduino`, 'success');
               
            } catch (error) {
                console.error('Detection error:', error);
                updateStatus('❌ Detection failed', 'error');
                document.getElementById('wasteCategoryLarge').textContent = 'ERROR';
                document.getElementById('wasteCategoryLarge').className = 'waste-category-large unknown';
            } finally {
                isDetecting = false;
                document.getElementById('infoPanel').classList.remove('detection-animation');
            }
        }
       
        async function sendWasteCommand(wasteCategory) {
            const commands = {
                'WET': 'w',
                'PAPER': 'p',
                'PLASTIC': 'l',
                'GLASS': 'g',
                'ELECTRONIC': 'e',
                'CAN': 'c',
                'UNKNOWN': 'u'
            };
           
            const command = commands[wasteCategory] || 'u';
           
            if (isArduinoConnected) {
                await sendToArduino(command);
            } else {
                logSerial('⚠️ Arduino not connected. Command not sent.');
            }
        }

        function processAIResults(classifications, detections, processingTime) {
            let detectedObject = 'unknown';
            let confidence = 0;

            if (detections.length > 0) {
                const topDetection = detections.reduce((prev, current) =>
                    (prev.score > current.score) ? prev : current
                );
                detectedObject = topDetection.class.toLowerCase();
                confidence = topDetection.score;
            } else if (classifications.length > 0) {
                const topClassification = classifications[0];
                detectedObject = topClassification.className.split(',')[0].toLowerCase();
                confidence = topClassification.probability;
            }

            let wasteCategory = 'UNKNOWN';
            const mappedCategory = wasteClassification[detectedObject];
           
            if (mappedCategory) {
                wasteCategory = mappedCategory.toUpperCase();
            } else {
                for (const [key, value] of Object.entries(wasteClassification)) {
                    if (detectedObject.includes(key) || key.includes(detectedObject)) {
                        wasteCategory = value.toUpperCase();
                        break;
                    }
                }
            }

            return {
                wasteCategory: wasteCategory,
                detectedObject: detectedObject,
                confidence: confidence,
                processingTime: processingTime
            };
        }

        function updateResults(result) {
            const wasteCategoryElement = document.getElementById('wasteCategoryLarge');
            wasteCategoryElement.textContent = result.wasteCategory;
           
            if (result.wasteCategory === 'WET') {
                wasteCategoryElement.className = 'waste-category-large wet';
            } else if (result.wasteCategory === 'PAPER') {
                wasteCategoryElement.className = 'waste-category-large paper';
            } else if (result.wasteCategory === 'PLASTIC') {
                wasteCategoryElement.className = 'waste-category-large plastic';
            } else if (result.wasteCategory === 'GLASS') {
                wasteCategoryElement.className = 'waste-category-large glass';
            } else if (result.wasteCategory === 'ELECTRONIC') {
                wasteCategoryElement.className = 'waste-category-large electronic';
            } else if (result.wasteCategory === 'CAN') {
                wasteCategoryElement.className = 'waste-category-large can';
            } else {
                wasteCategoryElement.className = 'waste-category-large unknown';
            }
           
            document.getElementById('confidenceValue').textContent = (result.confidence * 100).toFixed(0) + '%';
            document.getElementById('processingValue').textContent = result.processingTime.toFixed(2);
        }

        function updateStatistics(result) {
            totalDetections++;
            totalConfidence += result.confidence;
           
            document.getElementById('totalCount').textContent = totalDetections;
           
            const avgConfidence = (totalConfidence / totalDetections) * 100;
            document.getElementById('accuracyValue').textContent = avgConfidence.toFixed(0) + '%';
        }

        function updateStatus(message, type) {
            const statusIndicator = document.getElementById('statusIndicator');
            statusIndicator.textContent = message;
            statusIndicator.className = `status-indicator ${type}`;
        }

        // Handle page unload
        window.addEventListener('beforeunload', async () => {
            if (writer) {
                await writer.close();
            }
            if (serialPort) {
                await serialPort.close();
            }
        });