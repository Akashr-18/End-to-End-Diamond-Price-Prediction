import os, json, yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from DiamondRegressor import logger
import base64
import pickle
from sklearn.metrics import r2_score

@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    try:
        with open(path) as f:
            content = yaml.safe_load(f)
            logger.info(f"YAML file: {path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def read_json(path: Path) -> ConfigBox:
    try:
        with open(path) as f:
            content = json.load(f)
            logger.info(f"JSON file: {path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("json file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved successfully at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def decodeImage(imgString, fileName):
    image_data = base64.b64decode(imgString)
    with open(fileName, 'wb') as f:
        f.write(image_data)
        f.close()

def encodeImgae(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise e
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    except Exception as e:
        raise e