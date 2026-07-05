import os
import logging
from from_root import from_root
from logging.handlers import RotatingFileHandler
LOG_DIR = 'logs'

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# make the logs dir
os.makedirs(os.path.join(from_root(),LOG_DIR),exist_ok=True)

LOG_FILE_PATH = os.path.join(from_root(),LOG_DIR,'project_logs.log')

# formatter
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
)

file_handler = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=MAX_LOG_SIZE,
    backupCount=BACKUP_COUNT
)

file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# logger
logger = logging.getLogger("churn")
logger.setLevel(logging.DEBUG)
    
logger.addHandler(file_handler)
logger.addHandler(console_handler)