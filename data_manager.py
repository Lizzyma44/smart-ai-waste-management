import sqlite3
import json
import os
from datetime import datetime
import uuid

class DataManager:
    """
    Data management system for waste classification logs and storage
    """
    
    def __init__(self, db_path="waste_classification.db"):
        self.db_path = db_path
        self.image_dir = "captured_images"
        self.config_file = "config.json"
        
        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self.config = self._load_config()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create classifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                waste_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT,
                processing_time REAL,
                session_id TEXT
            )
        ''')
        
        # Create system_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Database initialized: {self.db_path}")
    
    def _load_config(self):
        """Load system configuration"""
        default_config = {
            "camera": {
                "resolution": [640, 480],
                "fps": 30,
                "device_index": 0
            },
            "classification": {
                "confidence_threshold": 0.5,
                "save_images": True,
                "auto_classify": True
            },
            "ui": {
                "theme": "light",
                "show_confidence": True,
                "show_processing_time": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def save_classification(self, waste_type, confidence, image_path=None, processing_time=0.0):
        """Save classification result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            session_id = str(uuid.uuid4())[:8]
            
            cursor.execute('''
                INSERT INTO classifications 
                (waste_type, confidence, image_path, processing_time, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (waste_type, confidence, image_path, processing_time, session_id))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error saving classification: {e}")
            return False
    
    def save_image(self, image, timestamp, waste_type=None):
        """Save captured image to disk"""
        try:
            filename = f"{timestamp}_{waste_type or 'unknown'}.jpg"
            filepath = os.path.join(self.image_dir, filename)
            
            import cv2
            cv2.imwrite(filepath, image)
            
            return filepath
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
    
    def get_classification_stats(self, days=7):
        """Get classification statistics for last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT waste_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM classifications
                WHERE timestamp > datetime('now', '-{} days')
                GROUP BY waste_type
            '''.format(days))
            
            results = cursor.fetchall()
            conn.close()
            
            stats = {}
            for waste_type, count, avg_confidence in results:
                stats[waste_type] = {
                    'count': count,
                    'avg_confidence': round(avg_confidence, 3)
                }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def get_recent_classifications(self, limit=10):
        """Get recent classification results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, waste_type, confidence, processing_time
                FROM classifications
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"Error getting recent classifications: {e}")
            return []
    
    def log_system_event(self, level, message, component=None):
        """Log system events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_logs (log_level, message, component)
                VALUES (?, ?, ?)
            ''', (level, message, component))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging system event: {e}")
    
    def get_config(self, section=None):
        """Get configuration section or entire config"""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, section, key, value):
        """Update configuration value"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        
        # Save to file
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False