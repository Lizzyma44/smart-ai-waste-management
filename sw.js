// Service Worker for Smart Waste Detection
const CACHE_NAME = 'smart-waste-detector-v1.0.0';
const urlsToCache = [
  '/',
  '/index.html',
  '/css/styles.css',
  '/js/app.js',
  '/js/camera.js',
  '/js/ai-detection.js',
  '/js/arduino-serial.js',
  '/manifest.json',
  // TensorFlow.js libraries (cached from CDN)
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0/dist/mobilenet.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.2/dist/coco-ssd.min.js',
  // Google Fonts
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
];

// Install event - cache resources
self.addEventListener('install', event => {
  console.log('SW: Installing service worker...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('SW: Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        console.log('SW: Installation complete');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('SW: Installation failed:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('SW: Activating service worker...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('SW: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('SW: Activation complete');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }
  
  // Skip Chrome extension requests
  if (event.request.url.startsWith('chrome-extension://')) {
    return;
  }
  
  // Skip data URLs
  if (event.request.url.startsWith('data:')) {
    return;
  }
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version if available
        if (response) {
          console.log('SW: Serving from cache:', event.request.url);
          return response;
        }
        
        // Fetch from network
        console.log('SW: Fetching from network:', event.request.url);
        return fetch(event.request).then(response => {
          // Don't cache non-successful responses
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          
          // Clone the response
          const responseToCache = response.clone();
          
          // Cache the fetched resource
          caches.open(CACHE_NAME)
            .then(cache => {
              cache.put(event.request, responseToCache);
            });
          
          return response;
        });
      })
      .catch(error => {
        console.error('SW: Fetch failed:', error);
        
        // Return offline page for navigation requests
        if (event.request.destination === 'document') {
          return caches.match('/index.html');
        }
        
        // For other requests, just fail
        throw error;
      })
  );
});

// Background sync for offline functionality
self.addEventListener('sync', event => {
  console.log('SW: Background sync triggered:', event.tag);
  
  if (event.tag === 'background-detection') {
    event.waitUntil(handleBackgroundDetection());
  }
});

async function handleBackgroundDetection() {
  try {
    // Handle queued detections when back online
    console.log('SW: Processing background detections...');
    
    // Get queued data from IndexedDB or localStorage
    const queuedDetections = await getQueuedDetections();
    
    for (const detection of queuedDetections) {
      try {
        // Process detection
        await processDetection(detection);
        
        // Remove from queue
        await removeFromQueue(detection.id);
        
      } catch (error) {
        console.error('SW: Failed to process detection:', error);
      }
    }
    
  } catch (error) {
    console.error('SW: Background sync failed:', error);
  }
}

// Notification handling
self.addEventListener('notificationclick', event => {
  console.log('SW: Notification clicked');
  
  event.notification.close();
  
  // Open the app
  event.waitUntil(
    clients.openWindow('/')
  );
});

// Push notifications (if needed in future)
self.addEventListener('push', event => {
  console.log('SW: Push message received');
  
  const options = {
    body: 'Waste detection completed!',
    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><text y="80" font-size="80">🤖</text></svg>',
    badge: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><text y="80" font-size="80">🔍</text></svg>',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Results',
        icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><text y="80" font-size="80">👁️</text></svg>'
      },
      {
        action: 'close',
        title: 'Close',
        icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><text y="80" font-size="80">❌</text></svg>'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('Smart Waste Detector', options)
  );
});

// Helper functions for offline functionality
async function getQueuedDetections() {
  // Implementation would use IndexedDB to store offline detections
  return [];
}

async function processDetection(detection) {
  // Process queued detection
  console.log('Processing detection:', detection);
}

async function removeFromQueue(id) {
  // Remove processed detection from queue
  console.log('Removing from queue:', id);
}

// Update available notification
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

// Share target (for future PWA sharing)
self.addEventListener('share', event => {
  console.log('SW: Share event received');
  
  event.waitUntil(
    // Handle shared content
    handleShare(event.data)
  );
});

async function handleShare(data) {
  // Handle shared images for waste detection
  console.log('Handling shared data:', data);
}

console.log('SW: Service worker script loaded');