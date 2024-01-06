import os
from pathlib import Path
from DiamondRegressor.entity.config_entity import (DataIngestionConfig, 
                                                   DataPreprocessingConfig,
                                                   ModelTrainingConfig,
                                                   ModelParameters)
from DiamondRegressor.constants import *
from DiamondRegressor.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url= config.source_url,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir= config.root_dir,
            data_file_path= config.data_file_path
        )

        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])
        create_directories([config.plot_dir])

        model_training_config = ModelTrainingConfig(
            root_dir= config.root_dir,
            plot_dir= config.plot_dir,
            plot_file_path = config.plot_file_path
        )
        return model_training_config
    
    def get_model_parameters(self) -> ModelParameters:
        params = self.params.models
        
        model_parameters_config = ModelParameters(
            xgboost_n_estimators = params.XgBoost.params.n_estimators,
            xgboost_learning_rate = params.XgBoost.params.learning_rate
        )
