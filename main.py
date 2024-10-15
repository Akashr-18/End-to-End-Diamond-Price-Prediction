from DiamondRegressor import logger
from DiamondRegressor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from DiamondRegressor.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from DiamondRegressor.pipeline.stage_03_model_training import ModelTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f'>>>>>>> Stage1: {STAGE_NAME} started <<<<<<<')
    data_ingestion_obj = DataIngestionTrainingPipeline()
    data_ingestion_obj.main()
    logger.info(f'>>>>>>> Stage1: {STAGE_NAME} completed <<<<<<<')   
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Preprocessing Stage'
try:
    logger.info(f'>>>>>>> Stage2: {STAGE_NAME} started <<<<<<<')
    data_preprocessing_obj = DataPreprocessingTrainingPipeline()
    train_arr, test_arr = data_preprocessing_obj.main()
    logger.info(f'>>>>>>> Stage2: {STAGE_NAME} completed <<<<<<<')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model Training Stage'
try:
    logger.info(f'>>>>>>> Stage3: {STAGE_NAME} started <<<<<<<')
    model_training_obj = ModelTrainingPipeline()
    model_training_obj.main(train_arr, test_arr)
    logger.info(f'>>>>>>> Stage3: {STAGE_NAME} completed <<<<<<<')
    
except Exception as e:
    logger.exception(e)
    raise e
