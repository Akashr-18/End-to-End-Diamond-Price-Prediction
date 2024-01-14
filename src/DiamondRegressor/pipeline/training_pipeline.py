from DiamondRegressor import logger
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.training import ModelTrainer
from DiamondRegressor.components.data_ingestion import DataIngestion
from DiamondRegressor.components.data_preprocessing import DataPreprocessing

STAGE_NAME = 'Training Pipleine'

class TrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_preprocessing_configuration = config.get_data_preprocessing_config()

        logger.info(f'>>>>>>> Stage1: Data Ingestion started <<<<<<<')
        data_ingestion_config = config.get_data_ingestion_config()
        data_preprocessing_config = config.get_data_preprocessing_config()
        training_configuration = config.get_training_config()
        logger.info(f'>>>>>>> Stage1: Data Ingestion completed <<<<<<<')

        logger.info(f'>>>>>>> Stage2: Data Pre-processing started <<<<<<<')
        data_ingestion = DataIngestion(config= data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.data_train_test_split()
        logger.info(f'>>>>>>> Stage2: Data Pre-processing completed <<<<<<<')

        logger.info(f'>>>>>>> Stage3: Model Training started <<<<<<<')
        data_preprocessor = DataPreprocessing(config = data_preprocessing_config,
                                       config_dataingestion = data_ingestion_config)
        train_arr, test_arr = data_preprocessor.data_preprocessing()
        logger.info(f'>>>>>>> Stage3: Model Training completed <<<<<<<')


        training = ModelTrainer(config = training_configuration)
        training.initate_model_training(train_arr, test_arr)

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<')
        obj = TrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e