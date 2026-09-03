"""
Author: samik1986
Date: 2026-09-03
"""
import logging
import os
from datetime import datetime

def get_logger(name, module_name="Global"):
    """
    Creates a centralized logger that outputs to both the console and a timestamped file.
    Logs are saved in the specific module's 'logs/' directory.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if instantiated repeatedly
    if logger.hasHandlers():
        return logger

    # Format
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    # Determine log directory based on the caller's module, or default to root logs/
    # If script is running from inside Neuro_Training, put it in Neuro_Training/logs/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if module_name == "Tokenization":
        log_dir = os.path.join(base_dir, 'Neuro_Tokenization', 'logs')
    elif module_name == "Model":
        log_dir = os.path.join(base_dir, 'Neuro_Model', 'logs')
    elif module_name == "Training":
        log_dir = os.path.join(base_dir, 'Neuro_Training', 'logs')
    elif module_name == "Retraining":
        log_dir = os.path.join(base_dir, 'Neuro_Retraining', 'logs')
    else:
        log_dir = os.path.join(base_dir, 'logs')
        
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.log")
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
