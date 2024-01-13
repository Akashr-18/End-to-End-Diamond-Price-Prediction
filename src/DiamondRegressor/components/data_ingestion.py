import os
import pandas as pd
from pathlib import Path
from DiamondRegressor import logger
import urllib.request as request
import zipfile
from sklearn.model_selection import train_test_split
from DiamondRegressor.utils.common import get_size
from DiamondRegressor.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"Downloading {filename}!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def data_train_test_split(self):
        data_folder_path = self.config.root_dir
        data_file_path = os.path.join(data_folder_path,'train_test.csv')
        data_file_path = Path(data_file_path)

        data = pd.read_csv(data_file_path)
        logger.info(f'Loaded df shape: {data.shape}')

        train_data, test_data = train_test_split(data, test_size=0.25)
        logger.info(f"Raw data converted into train data and test data of shape: {train_data.shape} and {test_data.shape}")

        train_data.to_csv(self.config.train_data_file_path,index=False)
        test_data.to_csv(self.config.test_data_file_path,index=False)
