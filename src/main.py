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
AIS_FILE_NAME = os.getenv('AIS_FILE_NAME')
NUMBER_OF_WORKERS = int(os.getenv('NUMBER_OF_WORKERS', 4))
SRC_PATH = os.getenv('SRC_PATH')


def run_main_processing():
    """Run the main vessel encounter processing."""
    logger.info("Starting main processing...")
    ve.vessel_encounters(AIS_FILE_NAME)


def create_training_data():
    """Create training data using AIS files."""
    logger.info("Creating training data...")
    AIS_data_info = helper.get_AIS_data_info()
    total_files = len(AIS_data_info)
    
    with ProcessPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        futures = [
            executor.submit(process_single_file, data, index + 1, total_files)
            for index, data in enumerate(AIS_data_info)
        ]
        
        for future in asyncio.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error during processing: {e}")


def process_single_file(data, current_index, total_files):
    """Process a single AIS data file."""
    logger.info(f"Processing file {current_index} out of {total_files}: {data['file_name']}")
    try:
        if helper.check_if_file_exists(data["file_name"]):
            logger.info(f"File {current_index} already processed: {data['file_name']}")
            return
        
        file_name_csv = helper.get_AIS_data_file(data["url"])
        ve.vessel_encounters(file_name_csv)
        file_path = os.path.join(SRC_PATH, file_name_csv)
        os.remove(file_path)
        logger.info(f"Successfully processed and removed file {current_index} out of {total_files}: {data['file_name']}")
    except Exception as e:
        logger.error(f"Error processing file {current_index} out of {total_files}: {data['file_name']}, Error: {e}")


def run_with_profiling():
    """Run the main function with profiling."""
    logger.info("Profiling enabled")
    with Profile() as pr:
        run_main_processing()
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.


def run_script(args):
    """Run the script based on the parsed arguments."""
    timestart = time.time()
    
    if args.profile:
        run_with_profiling()
    elif args.create_training_data:
        create_training_data()
    else:
        run_main_processing()
    
    timeend = time.time()
    logger.info(f"Total execution time: {timeend - timestart:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the risk assessment script with or without profiling.")
    parser.add_argument('--profile', action='store_true', help="Run with profiling")
    parser.add_argument('--create-training-data', action='store_true', help="Create training data for the ML model")
    args = parser.parse_args()
    
    run_script(args)
