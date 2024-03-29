from DiamondRegressor import logger
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.data_ingestion import DataIngestion


STAGE_NAME = 'Data Ingestion stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config= data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.data_train_test_split()
    
if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>> Stage1: {STAGE_NAME} started <<<<<<<')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage1: {STAGE_NAME} completed <<<<<<<')
        
    except Exception as e:
        logger.exception(e)
        raise e