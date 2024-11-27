import logging
import os
from dotenv import load_dotenv

load_dotenv()

LOGGER_FILE_PATH = os.getenv("LOGGER_FILE_PATH", "vessel_encounter.log")

def setup_logger(name):
    """
    Sets up a logger with the specified name, or returns an existing one if it exists.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Ensure no duplicate handlers are added
        logger.setLevel(logging.INFO)  # Default to INFO; can be adjusted as needed
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        # File handler
        file_handler = logging.FileHandler(LOGGER_FILE_PATH)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
