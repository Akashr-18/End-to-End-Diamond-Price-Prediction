from DiamondRegressor import logger
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.model_training import Training
from DiamondRegressor.components.model_finder import ModelFinder

STAGE_NAME = 'Model Training stage'

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_preprocessing_configuration = config.get_data_preprocessing_config()
        training_configuration = config.get_model_training_config()
        parameters_config = config.get_model_parameters_config()
        training = Training(config = data_preprocessing_configuration,
                 train_config = training_configuration)
        training.create_cluster_specific_models()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> Stage3: {STAGE_NAME} started <<<<<<<')
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage3: {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e
