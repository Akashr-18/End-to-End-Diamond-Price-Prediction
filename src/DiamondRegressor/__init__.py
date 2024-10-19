import logging
import sys
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging_dir = 'logs'
os.makedirs(logging_dir, exist_ok=True)
logging_filepath = os.path.join(logging_dir, "running_logs.log")

logging.basicConfig(
    level = logging.INFO, format = logging_str,
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logging_filepath)
    ])

logger = logging.getLogger('DiamnondRegressorLogger')
