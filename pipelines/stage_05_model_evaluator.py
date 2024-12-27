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
