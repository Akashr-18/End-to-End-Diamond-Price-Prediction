import pandas as pd
import os
from pathlib import Path
from DiamondRegressor import logger
from DiamondRegressor.entity.config_entity import DataPreprocessingConfig, DataIngestionConfig
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:

    def __init__(self, 
                 config: DataPreprocessingConfig,
                 config_dataingestion: DataIngestionConfig):
        
        self.config = config
        self.config_dataingestion = config_dataingestion

    def encode_categorical_data(self):

        data_folder_path = self.config_dataingestion.root_dir
        data_file_path = os.path.join(data_folder_path,'diamonds.csv')
        data_file_path = Path(data_file_path)

        df = pd.read_csv(data_file_path)
        df_updated = df.copy()
        logger.info(f'Loaded df shape: {df_updated.shape}')

        categorical_cols = df_updated.select_dtypes(include='object').columns
        numerical_cols = df_updated.select_dtypes(include='number').columns
        logger.info(f"Loaded data is having {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns")

        label_encoder = LabelEncoder()
        
        for col in categorical_cols:
            df_updated[col] = label_encoder.fit_transform(df_updated[col])
            logger.info(f"Encoding categorical column :{col}")
            logger.info(f"{col}: {df_updated[col].unique()}")
        logger.info(f"Encoding completed successfully")

        if len(df_updated.select_dtypes(include='object').columns) == 0:
            updated_data_file_path = self.config.data_file_path
            updated_data_file_path = Path(updated_data_file_path)
            df_updated.to_csv(updated_data_file_path, index=False)
            logger.info("Updated data file saved at {}".format(updated_data_file_path))

