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
    
    def train_cnn(self, model, X_train, y_train, model_name, epochs=50, batch_size=32):
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
