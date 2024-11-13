# src/utils/helpers.py

import os
import yaml
import logging
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

def load_config(config_path="src/config/config.yaml"):
    """
    Loads configuration from the specified YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        return {}

def setup_logging(log_level="INFO", log_file="src/logs/app.log"):
    """
    Sets up logging configuration.

    Args:
        log_level (str): The logging level, e.g., DEBUG, INFO, WARNING, ERROR.
        log_file (str): The file where logs should be saved.

    Returns:
        None
    """
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Log level: {log_level}, Log file: {log_file}")

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from environment variables.

    Returns:
        str: The OpenAI API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not found in environment variables.")
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
    return api_key
