from DiamondRegressor import logger
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.model_training import Training
from DiamondRegressor.components.model_finder import ModelFinder
from DiamondRegressor.components.training import ModelTrainer
import sys
sys.path.append('.')

STAGE_NAME = 'Model Training stage'

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self,train_arr,test_arr):
        config = ConfigurationManager()
        training_configuration = config.get_model_training_config()

        training = ModelTrainer(config = training_configuration)
        training.initate_model_training(train_arr, test_arr)

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> Stage3: {STAGE_NAME} started <<<<<<<')
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage3: {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e
