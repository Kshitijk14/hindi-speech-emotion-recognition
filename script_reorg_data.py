import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reorganize_dataset():
    try:
        # Define paths
        base_path = Path('artifacts/dataset')
        
        # Define emotion categories
        emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise']
        
        # Create emotion directories if they don't exist
        for emotion in emotions:
            emotion_dir = base_path / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for {emotion}")

        # Counter for moved files
        moved_files = 0
        
        # Walk through the current structure
        for session_dir in base_path.glob('*/session*'):
            if session_dir.is_dir():
                # Process each emotion folder in the session
                for emotion in emotions:
                    emotion_source = session_dir / emotion
                    if emotion_source.exists():
                        # Get all wav files
                        wav_files = list(emotion_source.glob('*.wav'))
                        
                        # Move each file to its corresponding emotion directory
                        for wav_file in wav_files:
                            # Create destination path
                            dest_path = base_path / emotion / f"{session_dir.parent.name}_{session_dir.name}_{wav_file.name}"
                            
                            # Move the file
                            shutil.copy2(wav_file, dest_path)
                            moved_files += 1
                            
                            logger.debug(f"Moved {wav_file.name} to {dest_path}")

        logger.info(f"Successfully moved {moved_files} files to their new locations")
        
        # Optional: Remove old directory structure
        # Uncomment these lines after verifying the reorganization was successful
        # for dir_name in ['1', '2', '3', '4', '5', '6', '7', '8']:
        #     old_dir = base_path / dir_name
        #     if old_dir.exists():
        #         shutil.rmtree(old_dir)
        #         logger.info(f"Removed old directory structure: {old_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting dataset reorganization...")
    reorganize_dataset()
    logger.info("Dataset reorganization completed!")