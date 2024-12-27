# src/config/config.py

import os
from pathlib import Path

class Config:
    def __init__(self):
        self.BASE_DIR = Path('../')
        self.DATA_DIR = self.BASE_DIR / 'artifacts/dataset'
        self.MODEL_DIR = self.BASE_DIR / 'models'
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

# src/data_ingestion/ingestion.py

import os
import glob
import librosa
import soundfile
import numpy as np
from pathlib import Path

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def extract_feature(self, file_name, **kwargs):
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

    def load_data(self):
        X, y = [], []
        dataset_path = str(self.config.DATA_DIR / '**/*.wav')

    for file_name in glob.glob(dataset_path, recursive=True):
            emotion_folder = os.path.basename(os.path.dirname(file_name))
            emotion = self.config.EMOTION_MAPPING.get(emotion_folder)

    if emotion not in self.config.AVAILABLE_EMOTIONS:
                continue

    features = self.extract_feature(file_name, mfcc=True, chroma=True, mel=True)
            X.append(features)
            y.append(emotion)

    return np.array(X), y

# src/data_preprocessing/preprocessor.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def preprocess(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

    # Convert to categorical for CNN
        y_train_cat = to_categorical(y_train_encoded)
        y_test_cat = to_categorical(y_test_encoded)

    # Reshape features for CNN
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_cnn': X_train_reshaped,
            'X_test_cnn': X_test_reshaped,
            'y_train_cnn': y_train_cat,
            'y_test_cnn': y_test_cat,
            'label_encoder': self.label_encoder
        }

# src/models/base_models.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

class BaseModels:
    @staticmethod
    def get_knn(n_neighbors=7):
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    @staticmethod
    def get_random_forest(n_estimators=150, random_state=42):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    @staticmethod
    def get_mlp(hidden_layer_sizes=(300,)):
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=0.01,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500
        )

    @staticmethod
    def get_cnn(input_shape, num_classes):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
        ])

    model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    return model

# src/training/model_trainer.py

import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_classical_model(self, model, X_train, y_train, model_name):
        model.fit(X_train, y_train)
        model_path = os.path.join(self.config.MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        return model

    def train_cnn(self, model, X_train, y_train, model_name, epochs=80, batch_size=32):
        checkpoint_path = os.path.join(self.config.MODEL_DIR, f"{model_name}.keras")

    callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
        ]

    history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    return model, history

# src/evaluation/model_evaluator.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_classical_model(self, model, X_test, y_test, model_name):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

    self._plot_confusion_matrix(cm, list(self.config.AVAILABLE_EMOTIONS), model_name)

    return {
            "model_name": model_name,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }

    def evaluate_cnn(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
        report = classification_report(y_test_classes, y_pred_classes)
        cm = confusion_matrix(y_test_classes, y_pred_classes)

    self._plot_confusion_matrix(cm, list(self.config.AVAILABLE_EMOTIONS), model_name)

    return {
            "model_name": model_name,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }

    def _plot_confusion_matrix(self, cm, labels, model_name):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# src/prediction/predictor.py

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

# src/main.py

from config.config import Config
from data_ingestion.ingestion import DataIngestion
from data_preprocessing.preprocessor import DataPreprocessor
from models.base_models import BaseModels
from training.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator
from prediction.predictor import Predictor

def main():
    # Initialize configuration
    config = Config()

    # Data ingestion
    data_ingestion = DataIngestion(config)
    X, y = data_ingestion.load_data()

    # Data preprocessing
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(X, y)

    # Initialize trainer and evaluator
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)

    # Train and evaluate KNN
    knn = BaseModels.get_knn()
    knn_model = trainer.train_classical_model(
        knn, processed_data['X_train'], processed_data['y_train'], 'knn'
    )
    knn_eval = evaluator.evaluate_classical_model(
        knn_model, processed_data['X_test'], processed_data['y_test'], 'KNN'
    )

    # Train and evaluate Random Forest
    rf = BaseModels.get_random_forest()
    rf_model = trainer.train_classical_model(
        rf, processed_data['X_train'], processed_data['y_train'], 'random_forest'
    )
    rf_eval = evaluator.evaluate_classical_model(
        rf_model, processed_data['X_test'], processed_data['y_test'], 'Random Forest'
    )

    # Train and evaluate MLP
    mlp = BaseModels.get_mlp()
    mlp_model = trainer.train_classical_model(
        mlp, processed_data['X_train'], processed_data['y_train'], 'mlp'
    )
    mlp_eval = evaluator.evaluate_classical_model(
        mlp_model, processed_data['X_test'], processed_data['y_test'], 'MLP'
    )

    # Train and evaluate CNN
    input_shape = (processed_data['X_train_cnn'].shape[1], 1)
    num_classes = len(config.AVAILABLE_EMOTIONS)
    cnn = BaseModels.get_cnn(input_shape, num_classes)
    cnn_model, history = trainer.train_cnn(
        cnn,
        processed_data['X_train_cnn'],
        processed_data['y_train_cnn'],
        'cnn'
    )
    cnn_eval = evaluator.evaluate_cnn(
        cnn_model,
        processed_data['X_test_cnn'],
        processed_data['y_test_cnn'],
        'CNN'
    )

    # Initialize predictor with best model
    predictor = Predictor(config, data_ingestion, processed_data['label_encoder'])

    return {
        'knn_eval': knn_eval,
        'rf_eval': rf_eval,
        'mlp_eval': mlp_eval,
        'cnn_eval': cnn_eval,
        'predictor': predictor
    }

if __name__ == "__main__":
    results = main()
