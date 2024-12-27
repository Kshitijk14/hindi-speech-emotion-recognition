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
