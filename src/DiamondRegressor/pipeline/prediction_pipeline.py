import os
import pandas as pd
import joblib
from pathlib import Path
from DiamondRegressor import logger
from DiamondRegressor.utils.common import load_object

class PredictPipeline:
    def __init__(self):
        logger.info("Constructing PredictPipeline")

    def predict(self, df_input):
        try:
            preprocessor_path = os.path.join("atrifacts", "preprocessing_object", "preprocessor.pkl")
            model_path = os.path.join("atrifacts", "models", "XGBRegressor.joblib")
            
            preprocessor_path = Path("artifacts\preprocessing_object\preprocessor.pkl")
            model_path = Path("artifacts\models\XGBRegressor.joblib")

            preprocessor=load_object(preprocessor_path)
            model = joblib.load(model_path)

            scaled_feature = preprocessor.transform(df_input)
            prediction  = model.predict(scaled_feature)

            return prediction
        except Exception as e:
            raise e

class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        try:
            input_data_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                }
            df = pd.DataFrame(input_data_dict)
            logger.info("DataFrame loaded")
            return df
        except Exception as e:
            raise e