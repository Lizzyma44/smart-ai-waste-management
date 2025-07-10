// AI Detection Module using TensorFlow.js
export class AIDetector {
    constructor() {
        this.mobileNetModel = null;
        this.cocoSsdModel = null;
        this.isLoaded = false;
        this.loadAttempts = 0;
        this.maxRetries = 3;
        this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        // Waste classification mapping
        this.wasteClassification = {
            // Wet/Organic Waste
            'banana': 'wet', 'apple': 'wet', 'orange': 'wet', 'broccoli': 'wet', 'carrot': 'wet',
            'hot dog': 'wet', 'pizza': 'wet', 'donut': 'wet', 'cake': 'wet', 'avocado': 'wet',
            'lemon': 'wet', 'sandwich': 'wet', 'food': 'wet', 'fruit': 'wet', 'vegetable': 'wet',
            'corn': 'wet', 'mushroom': 'wet', 'bread': 'wet', 'meat': 'wet', 'fish': 'wet',
            
            // Dry/Recyclable Waste
            'bottle': 'dry', 'wine glass': 'dry', 'cup': 'dry', 'knife': 'dry', 'spoon': 'dry',
            'bowl': 'dry', 'cell phone': 'dry', 'laptop': 'dry', 'mouse': 'dry', 'keyboard': 'dry',
            'book': 'dry', 'scissors': 'dry', 'can': 'dry', 'plastic': 'dry', 'glass': 'dry',
            'metal': 'dry', 'paper': 'dry', 'cardboard': 'dry', 'aluminum': 'dry', 'electronic': 'dry'
        };
    }
    
    async loadMobileNet(retryCount = 0) {
        try {
            console.log(`Loading MobileNet model... (attempt ${retryCount + 1}/${this.maxRetries})`);
            
            // Add timeout for mobile devices
            const timeoutMs = this.isMobile ? 30000 : 20000;
            const loadPromise = mobilenet.load();
            
            this.mobileNetModel = await Promise.race([
                loadPromise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('MobileNet loading timeout')), timeoutMs)
                )
            ]);
            
            console.log('✅ MobileNet loaded successfully');
            return true;
            
        } catch (error) {
            console.error(`Failed to load MobileNet (attempt ${retryCount + 1}):`, error);
            
            if (retryCount < this.maxRetries - 1) {
                console.log(`Retrying MobileNet load in 2 seconds...`);
                await new Promise(resolve => setTimeout(resolve, 2000));
                return this.loadMobileNet(retryCount + 1);
            }
            
            throw new Error(`Failed to load MobileNet after ${this.maxRetries} attempts: ${error.message}`);
        }
    }
    
    async loadCocoSSD(retryCount = 0) {
        try {
            console.log(`Loading COCO-SSD model... (attempt ${retryCount + 1}/${this.maxRetries})`);
            
            // Add timeout for mobile devices
            const timeoutMs = this.isMobile ? 45000 : 30000;
            
            // Mobile-optimized loading with smaller model if available
            const modelConfig = this.isMobile ? 
                { base: 'mobilenet_v1' } : 
                { base: 'mobilenet_v2' };
                
            const loadPromise = cocoSsd.load(modelConfig);
            
            this.cocoSsdModel = await Promise.race([
                loadPromise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('COCO-SSD loading timeout')), timeoutMs)
                )
            ]);
            
            console.log('✅ COCO-SSD loaded successfully');
            this.isLoaded = true;
            return true;
            
        } catch (error) {
            console.error(`Failed to load COCO-SSD (attempt ${retryCount + 1}):`, error);
            
            if (retryCount < this.maxRetries - 1) {
                console.log(`Retrying COCO-SSD load in 3 seconds...`);
                await new Promise(resolve => setTimeout(resolve, 3000));
                return this.loadCocoSSD(retryCount + 1);
            }
            
            throw new Error(`Failed to load COCO-SSD after ${this.maxRetries} attempts: ${error.message}`);
        }
    }
    
    async classifyWaste(imageElement) {
        if (!this.isLoaded) {
            throw new Error('AI models not loaded. Please refresh the page to retry loading.');
        }
        
        const startTime = performance.now();
        
        try {
            // Run both models in parallel for better accuracy
            const [classifications, detections] = await Promise.all([
                this.classifyWithMobileNet(imageElement),
                this.detectWithCocoSSD(imageElement)
            ]);
            
            const result = this.processResults(classifications, detections);
            const processingTime = (performance.now() - startTime) / 1000;
            
            return {
                ...result,
                processingTime
            };
            
        } catch (error) {
            console.error('Classification error:', error);
            
            // Provide fallback basic classification
            if (error.message.includes('not loaded')) {
                throw error;
            }
            
            // Return basic fallback result for processing errors
            return {
                wasteCategory: 'UNKNOWN',
                object: 'unidentified',
                confidence: 0,
                source: 'fallback',
                processingTime: 0
            };
        }
    }
    
    // Add method to check TensorFlow.js availability
    static async checkTensorFlowSupport() {
        try {
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded');
            }
            
            if (typeof mobilenet === 'undefined') {
                throw new Error('MobileNet not loaded');
            }
            
            if (typeof cocoSsd === 'undefined') {
                throw new Error('COCO-SSD not loaded');
            }
            
            // Test WebGL support
            const webglSupported = tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE');
            
            return {
                supported: true,
                webgl: webglSupported,
                backend: tf.getBackend()
            };
            
        } catch (error) {
            return {
                supported: false,
                error: error.message,
                webgl: false,
                backend: 'none'
            };
        }
    }
    
    async classifyWithMobileNet(imageElement) {
        try {
            if (!this.mobileNetModel) {
                throw new Error('MobileNet model not loaded');
            }
            
            // Add timeout for mobile processing
            const timeoutMs = this.isMobile ? 10000 : 5000;
            const classifyPromise = this.mobileNetModel.classify(imageElement);
            
            const predictions = await Promise.race([
                classifyPromise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('MobileNet classification timeout')), timeoutMs)
                )
            ]);
            
            return predictions || [];
            
        } catch (error) {
            console.error('MobileNet classification error:', error);
            return [];
        }
    }
    
    async detectWithCocoSSD(imageElement) {
        try {
            if (!this.cocoSsdModel) {
                throw new Error('COCO-SSD model not loaded');
            }
            
            // Add timeout for mobile processing
            const timeoutMs = this.isMobile ? 15000 : 10000;
            const detectPromise = this.cocoSsdModel.detect(imageElement);
            
            const predictions = await Promise.race([
                detectPromise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('COCO-SSD detection timeout')), timeoutMs)
                )
            ]);
            
            return predictions || [];
            
        } catch (error) {
            console.error('COCO-SSD detection error:', error);
            return [];
        }
    }
    
    processResults(classifications, detections) {
        let detectedObject = 'unknown';
        let confidence = 0;
        let source = 'none';
        
        // Prefer COCO-SSD detections (more accurate for objects)
        if (detections.length > 0) {
            const topDetection = detections.reduce((prev, current) => 
                (prev.score > current.score) ? prev : current
            );
            
            if (topDetection.score > 0.3) {
                detectedObject = topDetection.class;
                confidence = topDetection.score;
                source = 'coco-ssd';
            }
        }
        
        // Fallback to MobileNet if COCO-SSD didn't find anything confident
        if (confidence < 0.3 && classifications.length > 0) {
            const topClassification = classifications[0];
            if (topClassification.probability > 0.1) {
                detectedObject = topClassification.className;
                confidence = topClassification.probability;
                source = 'mobilenet';
            }
        }
        
        // Determine waste category
        const wasteCategory = this.determineWasteCategory(detectedObject);
        
        return {
            detectedObject,
            wasteCategory,
            confidence,
            source,
            allClassifications: classifications,
            allDetections: detections
        };
    }
    
    determineWasteCategory(detectedObject) {
        // Direct mapping
        if (this.wasteClassification[detectedObject]) {
            const category = this.wasteClassification[detectedObject];
            return category === 'wet' ? 'WET WASTE' : 'DRY WASTE';
        }
        
        // Fuzzy matching
        const objectLower = detectedObject.toLowerCase();
        
        for (const [key, value] of Object.entries(this.wasteClassification)) {
            if (objectLower.includes(key) || key.includes(objectLower)) {
                return value === 'wet' ? 'WET WASTE' : 'DRY WASTE';
            }
        }
        
        // Keyword-based classification
        const wetKeywords = [
            'food', 'fruit', 'vegetable', 'organic', 'eat', 'edible', 'meal',
            'snack', 'fresh', 'ripe', 'cooked', 'baked', 'fried', 'raw',
            'meat', 'fish', 'dairy', 'cheese', 'milk', 'egg', 'bread',
            'grain', 'rice', 'pasta', 'soup', 'sauce', 'juice', 'coffee',
            'tea', 'plant', 'leaf', 'stem', 'root', 'seed', 'peel', 'skin'
        ];
        
        const dryKeywords = [
            'bottle', 'can', 'glass', 'plastic', 'metal', 'paper', 'cardboard',
            'aluminum', 'steel', 'tin', 'container', 'package', 'wrapper',
            'bag', 'box', 'electronic', 'device', 'gadget', 'appliance',
            'tool', 'instrument', 'utensil', 'equipment', 'machinery',
            'clothing', 'fabric', 'textile', 'synthetic', 'artificial',
            'manufactured', 'processed', 'industrial', 'chemical'
        ];
        
        // Check wet keywords
        if (wetKeywords.some(keyword => objectLower.includes(keyword))) {
            return 'WET WASTE';
        }
        
        // Check dry keywords
        if (dryKeywords.some(keyword => objectLower.includes(keyword))) {
            return 'DRY WASTE';
        }
        
        // Advanced heuristics based on object characteristics
        return this.advancedClassification(detectedObject, objectLower);
    }
    
    advancedClassification(detectedObject, objectLower) {
        // Common dry waste patterns
        const dryPatterns = [
            /bottle/i, /can/i, /glass/i, /plastic/i, /metal/i, /paper/i,
            /card/i, /box/i, /container/i, /wrapper/i, /package/i,
            /electronic/i, /device/i, /phone/i, /computer/i, /tool/i
        ];
        
        // Common wet waste patterns
        const wetPatterns = [
            /apple/i, /banana/i, /orange/i, /fruit/i, /vegetable/i,
            /food/i, /meat/i, /fish/i, /bread/i, /cake/i, /pizza/i,
            /organic/i, /fresh/i, /cooked/i, /edible/i
        ];
        
        // Check patterns
        if (wetPatterns.some(pattern => pattern.test(objectLower))) {
            return 'WET WASTE';
        }
        
        if (dryPatterns.some(pattern => pattern.test(objectLower))) {
            return 'DRY WASTE';
        }
        
        // Default classification based on common sense
        // Most unidentified objects in waste context are likely dry waste
        return 'DRY WASTE';
    }
    
    // Get model information
    getModelInfo() {
        return {
            mobileNetLoaded: !!this.mobileNetModel,
            cocoSsdLoaded: !!this.cocoSsdModel,
            isReady: this.isLoaded,
            wasteCategories: Object.keys(this.wasteClassification).length
        };
    }
    
    // Preprocess image for better accuracy
    preprocessImage(imageElement, targetSize = 224) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = targetSize;
        canvas.height = targetSize;
        
        // Draw and resize image
        ctx.drawImage(imageElement, 0, 0, targetSize, targetSize);
        
        return canvas;
    }
    
    // Get confidence level description
    getConfidenceLevel(confidence) {
        if (confidence >= 0.8) return 'Very High';
        if (confidence >= 0.6) return 'High';
        if (confidence >= 0.4) return 'Medium';
        if (confidence >= 0.2) return 'Low';
        return 'Very Low';
    }
    
    // Analyze detection quality
    analyzeDetectionQuality(result) {
        const analysis = {
            confidence: result.confidence,
            confidenceLevel: this.getConfidenceLevel(result.confidence),
            source: result.source,
            reliability: 'unknown'
        };
        
        // Determine reliability
        if (result.source === 'coco-ssd' && result.confidence > 0.6) {
            analysis.reliability = 'high';
        } else if (result.source === 'mobilenet' && result.confidence > 0.3) {
            analysis.reliability = 'medium';
        } else {
            analysis.reliability = 'low';
        }
        
        return analysis;
    }
}