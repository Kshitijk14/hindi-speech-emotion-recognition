import os
import glob
import librosa
import soundfile
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config
        
    def extract_feature(self, file_name, **kwargs):
        try:
            mfcc = kwargs.get("mfcc")
            chroma = kwargs.get("chroma")
            mel = kwargs.get("mel")
            
            with soundfile.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate
                
                if chroma:
                    stft = np.abs(librosa.stft(X))
                    
                result = np.array([])
                
                if mfcc:
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                    result = np.hstack((result, mfccs))
                
                if chroma:
                    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, chroma))
                    
                if mel:
                    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, mel))
                    
                return result
        except Exception as e:
            logger.error(f"Error extracting features from {file_name}: {str(e)}")
            return None
    
    def load_data(self):
        X, y = [], []
        try:
            # Print current working directory and data directory path
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Looking for data in: {self.config.DATA_DIR}")
            
            # Get all WAV files recursively
            dataset_path = str(self.config.DATA_DIR / '**' / '*.wav')
            wav_files = glob.glob(dataset_path, recursive=True)
            
            if not wav_files:
                logger.error(f"No WAV files found in {dataset_path}")
                raise ValueError("No audio files found in the specified directory")
            
            logger.info(f"Found {len(wav_files)} WAV files")
            
            for file_name in wav_files:
                try:
                    # Extract emotion from directory structure
                    emotion_folder = os.path.basename(os.path.dirname(file_name))
                    emotion = self.config.EMOTION_MAPPING.get(emotion_folder)
                    
                    if emotion not in self.config.AVAILABLE_EMOTIONS:
                        logger.warning(f"Skipping file {file_name} - emotion {emotion_folder} not in available emotions")
                        continue
                    
                    # Extract features
                    features = self.extract_feature(file_name, mfcc=True, chroma=True, mel=True)
                    
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")
                    continue
            
            if not X or not y:
                raise ValueError("No features were successfully extracted from the audio files")
            
            logger.info(f"Successfully loaded {len(X)} samples")
            return np.array(X), y
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise