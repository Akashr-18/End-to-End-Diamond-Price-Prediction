import os
import joblib
from DiamondRegressor import logger
from DiamondRegressor.utils.common import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from DiamondRegressor.entity.config_entity import DataPreprocessingConfig, DataIngestionConfig, TrainingConfig
from DiamondRegressor.components.model_finder import ModelFinder

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def initate_model_training(self,train_array,test_array):
        logger.info('Splitting Dependent and Independent variables from train and test data')
        X_train, y_train, X_test, y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )

        model, model_name = ModelFinder.find_best_model(X_train, X_test, y_train, y_test)
        logger.info(f" Best model is {model_name}")
        model_foder = self.config.root_dir
        file_name = f"{model_name}.joblib"
        model_path = os.path.join(model_foder, file_name) 
        joblib.dump(model, model_path)
        logger.info(f"Model successfully saved at {model_path}")
        

   