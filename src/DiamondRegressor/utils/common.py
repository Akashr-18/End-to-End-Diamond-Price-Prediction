import os, json, yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from DiamondRegressor import logger
import base64

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