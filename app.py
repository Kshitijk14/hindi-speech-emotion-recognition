import os
import librosa
import sounddevice as sd
import wavio
import tempfile
from scipy.io import wavfile
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st

# Import pipeline components
from config.config import Config
from pipelines.stage_01_ingestion import DataIngestion
from pipelines.stage_02_preprocessor import DataPreprocessor
from pipelines.stage_06_predictor import Predictor


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

model_path_dir = 'models/cnn.keras'


class AudioProcessor:
    def __init__(self):
        self.config = Config()
        self.data_ingestion = DataIngestion(self.config)
        self.preprocessor = DataPreprocessor()
        
    def process_audio(self, audio_path):
        # Use the same feature extraction from DataIngestion
        features = self.data_ingestion.extract_feature(
            audio_path,
            mfcc=True,
            chroma=True,
            mel=True
        )
        
        if features is None:
            raise ValueError("Failed to extract features from the audio file")
            
        # Reshape for CNN (adding batch dimension)
        features = features.reshape(1, features.shape[0], 1)
        
        return features

def main():
    st.title("Audio Emotion Detection")
    st.write("Upload an audio file or record your voice to detect emotions!")

    try:
        # Initialize components
        processor = AudioProcessor()
        model = load_model(model_path_dir)
        
        # Get number of features for verification
        dummy_features = processor.data_ingestion.extract_feature(
            next(processor.config.DATA_DIR.glob('**/*.wav')),
            mfcc=True,
            chroma=True,
            mel=True
        )
        st.sidebar.write(f"Expected features shape: {dummy_features.shape}")
        
        # Print model summary for debugging
        # st.sidebar.write("Model Summary:")
        # model.summary(print_fn=lambda x: st.sidebar.text(x))
        
        # Define emotion labels
        emotions = processor.config.AVAILABLE_EMOTIONS
        
        # File upload
        uploaded_file = st.file_uploader("Choose an audio file...", type=['wav'])
        
        # Record audio
        if st.button('Record Audio'):
            st.write("🎙️ Recording... (3 seconds)")
            duration = 3  # seconds
            recording = sd.rec(int(duration * processor.config.SAMPLE_RATE), 
                            samplerate=processor.config.SAMPLE_RATE, 
                            channels=1)
            sd.wait()
            
            # Save recording to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                wavio.write(tmp_file.name, recording, processor.config.SAMPLE_RATE, sampwidth=2)
                audio_path = tmp_file.name
                
            try:
                # Process and predict
                features = processor.process_audio(audio_path)
                st.sidebar.write(f"Input features shape: {features.shape}")  # Debug info
                prediction = model.predict(features)
                predicted_emotion = emotions[np.argmax(prediction)]
                
                # Display results
                st.success(f"Predicted emotion: {predicted_emotion}")
                st.audio(audio_path)
                
            finally:
                # Clean up temporary file
                os.unlink(audio_path)
            
        # Handle uploaded file
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            try:
                # Process and predict
                features = processor.process_audio(audio_path)
                st.sidebar.write(f"Input features shape: {features.shape}")  # Debug info
                prediction = model.predict(features)
                predicted_emotion = emotions[np.argmax(prediction)]
                
                # Display results
                st.success(f"Predicted emotion: {predicted_emotion}")
                st.audio(uploaded_file)
                
                # Show prediction probabilities
                st.write("Prediction probabilities:")
                probs_df = pd.DataFrame({
                    'Emotion': emotions,
                    'Probability': prediction[0] * 100
                })
                st.bar_chart(probs_df.set_index('Emotion'))
                
            finally:
                # Clean up temporary file
                os.unlink(audio_path)
            
    except Exception as e:
        st.error(f"Error loading model or processing audio: {str(e)}")
        st.write("Please check the error and make sure all components are correctly configured")
        st.write(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()