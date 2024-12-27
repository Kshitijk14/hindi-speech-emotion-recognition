# src/config/config.py
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # Get the absolute path of the current file
        current_file = Path(__file__).resolve()
        
        # Find project root by looking for the repository directory
        repo_name = "hindi-speech-emotion-recognition"
        current_path = current_file.parent
        while current_path.name != repo_name and current_path.parent != current_path:
            current_path = current_path.parent
            
        if current_path.name != repo_name:
            raise RuntimeError(f"Could not find project root directory: {repo_name}")
            
        self.BASE_DIR = current_path
        self.DATA_DIR = self.BASE_DIR / 'artifacts' / 'dataset'
        self.MODEL_DIR = self.BASE_DIR / 'models'
        
        logger.info(f"Base directory: {self.BASE_DIR}")
        logger.info(f"Data directory: {self.DATA_DIR}")
        logger.info(f"Model directory: {self.MODEL_DIR}")
        
        self.AVAILABLE_EMOTIONS = {
            "angry", "disgust", "afraid", "happy", 
            "calm", "sad", "sarcastic", "surprise"
        }
        self.EMOTION_MAPPING = {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "afraid",
            "happy": "happy",
            "neutral": "calm",
            "sad": "sad",
            "sarcastic": "sarcastic",
            "surprise": "surprise",
        }
        
        # Create directories if they don't exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
        # Create data directory structure if it doesn't exist
        if not self.DATA_DIR.exists():
            logger.warning(f"Data directory not found: {self.DATA_DIR}")
            logger.info("Creating required directory structure...")
            
            try:
                # Create main data directories
                os.makedirs(self.DATA_DIR, exist_ok=True)
                
                # Create emotion subdirectories
                for emotion in self.EMOTION_MAPPING.keys():
                    os.makedirs(self.DATA_DIR / emotion, exist_ok=True)
                
                logger.info("Directory structure created successfully!")
                logger.info("Please move your audio files into the appropriate emotion folders in:")
                logger.info(f"{self.DATA_DIR}")
                
            except Exception as e:
                logger.error(f"Error creating directory structure: {str(e)}")
                raise