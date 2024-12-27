# verify_structure.py
import os
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_and_setup_structure():
    # Get project root directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    
    # Find project root
    while project_root.name != "hindi-speech-emotion-recognition" and project_root.parent != project_root:
        project_root = project_root.parent
    
    if project_root.name != "hindi-speech-emotion-recognition":
        raise RuntimeError("Could not find project root directory")
    
    # Define required paths
    artifacts_dir = project_root / 'artifacts'
    dataset_dir = artifacts_dir / 'dataset'
    models_dir = project_root / 'models'
    
    # Create required directories
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Models directory: {models_dir}")
    
    return project_root, dataset_dir

if __name__ == "__main__":
    try:
        project_root, dataset_dir = verify_and_setup_structure()
        logger.info("Directory structure verified and set up successfully!")
        
        # Print instructions
        print("\nNext steps:")
        print("1. Run the dataset reorganization script to organize your audio files")
        print("2. Verify that the audio files are correctly placed in their emotion folders")
        print("3. Run your main.py script again")
        
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")