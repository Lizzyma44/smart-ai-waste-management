// AI Detection Module using TensorFlow.js
export class AIDetector {
    constructor() {
        this.mobileNetModel = null;
        this.cocoSsdModel = null;
        this.isLoaded = false;
        
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
    
    async loadMobileNet() {
        try {
            console.log('Loading MobileNet model...');
            this.mobileNetModel = await mobilenet.load();
            console.log('✅ MobileNet loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load MobileNet:', error);
            throw error;
        }
    }
    
    async loadCocoSSD() {
        try {
            console.log('Loading COCO-SSD model...');
            this.cocoSsdModel = await cocoSsd.load();
            console.log('✅ COCO-SSD loaded successfully');
            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('Failed to load COCO-SSD:', error);
            throw error;
        }
    }
    
    async classifyWaste(imageElement) {
        if (!this.isLoaded) {
            throw new Error('AI models not loaded');
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
            throw error;
        }
    }
    
    async classifyWithMobileNet(imageElement) {
        try {
            const predictions = await this.mobileNetModel.classify(imageElement);
            return predictions;
        } catch (error) {
            console.error('MobileNet classification error:', error);
            return [];
        }
    }
    
    async detectWithCocoSSD(imageElement) {
        try {
            const predictions = await this.cocoSsdModel.detect(imageElement);
            return predictions;
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