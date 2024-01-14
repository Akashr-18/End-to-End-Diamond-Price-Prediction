from DiamondRegressor import logger
from DiamondRegressor.utils.common import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from DiamondRegressor.entity.config_entity import DataPreprocessingConfig, DataIngestionConfig, TrainingConfig

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

        models={
        'LinearRegression':LinearRegression(),
        'Lasso':Lasso(),
        'Ridge':Ridge(),
        'Elasticnet':ElasticNet()
    }
        
        model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
        print(model_report)
        print('\n====================================================================================\n')
        logger.info(f'Model Report : {model_report}')

        # To get best model score from dictionary 
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        best_model = models[best_model_name]

        print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
        print('\n====================================================================================\n')
        logger.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

        save_object(
                file_path=self.config.model_path,
                obj=best_model
        )
          