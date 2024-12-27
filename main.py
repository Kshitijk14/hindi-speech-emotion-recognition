from config.config import Config
from pipelines.stage_01_ingestion import DataIngestion
from pipelines.stage_02_preprocessor import DataPreprocessor
from pipelines.stage_03_base_models import BaseModels
from pipelines.stage_04_model_trainer import ModelTrainer
from pipelines.stage_05_model_evaluator import ModelEvaluator
from pipelines.stage_06_predictor import Predictor

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