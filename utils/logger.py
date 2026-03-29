import logging
import os
import sys
from datetime import datetime

# Import config directly for log directory
from utils.config import LOG_DIR

def setup_logger(name="haryanvi_slm_pipeline"):
    # Ensure logs folder exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}.log"))
    
    # Create formatters and add it to handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    c_format = logging.Formatter(format_str)
    f_format = logging.Formatter(format_str)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

logger = setup_logger()
