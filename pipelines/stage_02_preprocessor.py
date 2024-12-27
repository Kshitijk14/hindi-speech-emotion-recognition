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
