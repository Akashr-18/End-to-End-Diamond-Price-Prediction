from DiamondRegressor import logger
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.data_ingestion import DataIngestion
from DiamondRegressor.components.data_preprocessing import DataPreprocessing

STAGE_NAME = 'Data Preprocessing stage'

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessor = DataPreprocessing(config = data_preprocessing_config,
                                       config_dataingestion = data_ingestion_config)
        train_arr, test_arr = data_preprocessor.data_preprocessing()
        return train_arr, test_arr

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> Stage2: {STAGE_NAME} started <<<<<<<')
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage2: {STAGE_NAME} completed <<<<<<<')
        
    except Exception as e:
        logger.exception(e)
        raise e