# src/utils/logger.py

import logging
import os
from src.utils.helpers import load_config

def setup_logger():
    """
    Sets up a logger with configuration settings from the config file.
    Ensures a single, consistent logger instance across the application.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Load logging configuration
    config = load_config()
    log_level = config["logging"].get("level", "INFO")
    log_file = config["logging"].get("log_file", "src/logs/app.log")

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Create a logger instance
    logger = logging.getLogger("CodeAndReviewLogger")
    logger.setLevel(log_level)
    logger.info(f"Logger initialized with level {log_level}, logging to {log_file}")
    return logger

# Initialize and retrieve a single logger instance
logger = setup_logger()
