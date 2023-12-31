from DiamondRegressor import logger
from DiamondRegressor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = 'Data Ingestion stage'
try:
    logger.info(f'>>>>>>> Stage1: {STAGE_NAME} started <<<<<<<')
    data_ingestion_obj = DataIngestionTrainingPipeline()
    data_ingestion_obj.main()
    logger.info(f'>>>>>>> Stage1: {STAGE_NAME} completed <<<<<<<')
        
except Exception as e:
    logger.exception(e)
    raise e