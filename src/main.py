import logging
from cProfile import Profile
from pstats import Stats
from dotenv import load_dotenv
import asyncio
import time
import encounters.vessel_encounter as ve
import pandas as pd
import os
import argparse
import encounters.helper as helper
from concurrent.futures import ProcessPoolExecutor
from utils.logger import setup_logger


# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logger(__name__) 

# Constants
PATH_TO_AIS_FILE = os.getenv('PATH_TO_AIS_FILE')
NUMBER_OF_WORKERS = int(os.getenv('NUMBER_OF_WORKERS', 4))
SRC_PATH = os.getenv('SRC_PATH')

def main():
    logger.info("Starting main processing...")
    ve.vessel_encounters(PATH_TO_AIS_FILE)

def get_training_data():
    logger.info("Creating training data...")
    AIS_data_info = helper.get_AIS_data_info()
        
    # Define a thread pool
    with ProcessPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        futures = [
            executor.submit(run_vessel_encounter_for_url, data)
            for data in AIS_data_info
        ]
        
        for future in asyncio.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error during processing: {e}")

def run_vessel_encounter_for_url(data):
    logger.info(f"Running vessel encounter for date: {data['file_name']}")
    try:
        if helper.check_if_file_exists(data["file_name"]):
            logger.info("File already processed")
            return
        
        file_name_csv = helper.get_AIS_data_file(data["url"])
        ve.vessel_encounters(file_name_csv)
        file_path = os.path.join(SRC_PATH, file_name_csv)
        os.remove(file_path)
        logger.info(f"Successfully processed and removed file: {data['file_name']}")
    except Exception as e:
        logger.error(f"Error processing {data['file_name']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the risk assessment script with or without profiling.")
    parser.add_argument('--profile', action='store_true', help="Run with profiling")
    parser.add_argument('--create-training-data', action='store_true', help="Create training data for the ML model")
    args = parser.parse_args()

    if args.profile:
        logger.info("Profiling enabled")
        with Profile() as pr:
            main()
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.
    elif args.create_training_data:
        timestart = time.time()
        get_training_data()
        timeend = time.time()
        logger.info(f"Time taken to create training data: {timeend - timestart} seconds")
    else:
        timestart = time.time()
        #main()
        get_training_data()
        timeend = time.time()
        logger.info(f"Time taken for main processing: {timeend - timestart} seconds")
