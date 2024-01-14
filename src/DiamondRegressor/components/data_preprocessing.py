import pandas as pd
import numpy as np
import os
# from pathlib import Path
from DiamondRegressor import logger
from DiamondRegressor.utils.common import save_object
from DiamondRegressor.entity.config_entity import DataPreprocessingConfig, DataIngestionConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer #Handling missing values
from sklearn.preprocessing import OrdinalEncoder #Ordinal Encoding
from sklearn.preprocessing import StandardScaler #Feature Scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreprocessing:

    def __init__(self, 
                 config: DataPreprocessingConfig,
                 config_dataingestion: DataIngestionConfig):
        
        self.config = config
        self.config_dataingestion = config_dataingestion

    def data_transformation(self):

        dataframe_path = self.config_dataingestion.train_data_file_path
        df = pd.read_csv(dataframe_path)
        logger.info(f'For data transformation loaded sample df shape: {df.shape}')

        target_column_name = 'price'
        df = df.drop(columns=['id',target_column_name], axis=1)

        # categorical_cols = ['cut', 'color','clarity']
        # numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
        
        categorical_columns = df.select_dtypes(include = 'object').columns
        numerical_columns = df.select_dtypes(exclude = 'object').columns

        categorical_columns = list(categorical_columns)
        numerical_columns = list(numerical_columns)
        logger.info(f"Categorical columns: {categorical_columns}")
        logger.info(f"Numerical columns: {numerical_columns}")

        cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
        color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

        num_pipeline = Pipeline(
            steps = [
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoding", OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numerical pipeline", num_pipeline, numerical_columns),
                ("categorical pipeline", cat_pipeline, categorical_columns)
            ]
        )
        return preprocessor  

    def data_preprocessing(self):

        train_data_file_path = self.config_dataingestion.train_data_file_path
        test_data_file_path = self.config_dataingestion.test_data_file_path

        train_df = pd.read_csv(train_data_file_path)
        test_df = pd.read_csv(test_data_file_path)
        logger.info(f"Train data file and Test data file loaded")

        preprocessing_obj = self.data_transformation()

        target_column_name = 'price'

        X_train_df = train_df.drop(columns = [target_column_name, 'id'], axis=1)
        y_train_df = train_df[[target_column_name]]

        X_test_df = test_df.drop(columns = [target_column_name, 'id'], axis=1)
        y_test_df = test_df[[target_column_name]]

        X_train_arr = preprocessing_obj.fit_transform(X_train_df)
        X_test_arr = preprocessing_obj.transform(X_test_df)

        logger.info("Applying preprocessing object on training and testing datasets.")

        train_arr = np.c_[X_train_arr, np.array(y_train_df)]
        test_arr = np.c_[X_test_arr, np.array(y_test_df)]
        # print(train_arr.shape)
        # print("######################")
        # print(test_arr[0])

        save_object(
            file_path=self.config.preprocessor_file_path,
            obj=preprocessing_obj
        )
        
        logger.info("preprocessing pickle file saved")
        
        return (
            train_arr,
            test_arr
        )

    