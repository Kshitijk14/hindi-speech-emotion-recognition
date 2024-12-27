import numpy as np
import joblib
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, config, data_ingestion, label_encoder):
        self.config = config
        self.data_ingestion = data_ingestion
        self.label_encoder = label_encoder
        
    def load_model(self, model_path):
        if model_path.endswith('.keras'):
            return load_model(model_path)
        else:
            return joblib.load(model_path)
    
    def predict_audio(self, audio_path, model):
        features = self.data_ingestion.extract_feature(
            audio_path, mfcc=True, chroma=True, mel=True
        )
        
        if isinstance(model, str):
            model = self.load_model(model)
        
        if model_path.endswith('.keras'):
            features = features.reshape(1, -1, 1)
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction)
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        else:
            features = features.reshape(1, -1)
            emotion = model.predict([features])[0]
        
        return emotion
