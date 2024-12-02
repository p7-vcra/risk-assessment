import asyncio
import time
import os
import argparse
import pandas as pd
import encounters.helper as helper
import encounters.vessel_encounter as ve

from concurrent.futures import ProcessPoolExecutor
from cProfile import Profile
from pstats import Stats
from vcra import training, vessel_cri
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.getenv('DATA_DIR')
AIS_FILE_NAME = os.getenv('AIS_FILE_NAME')
NUMBER_OF_WORKERS = int(os.getenv('NUMBER_OF_WORKERS', 4))

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
        file_path = os.path.join(DATA_DIR, file_name_csv)
        logger.info(f"Attempting to remove file {file_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Run the risk assessment scripts")
    subparsers = parser.add_subparsers(
        dest="action", help="Specify the type of script to run"
    )

    parser_encounters = subparsers.add_parser(
        "encounters", help="Run the vessel encounters script"
    )
    parser_encounters.add_argument(
        "--profile", "-p", action="store_true", help="Run with profiling"
    )
    parser_encounters.add_argument('--create-training-data', action='store_true', help="Create training data for the ML model")

    parser_training = subparsers.add_parser("training", help="Run the training script")
    parser_training.add_argument(
        "--use-checkpoint",
        "-uc",
        action="store_true",
        help="Flag to use checkpoint for resuming training",
    )
    parser_training.add_argument(
        "--sample-data",
        "-sd",
        type=int,
        help="Number of sample data points to generate for training",
    )

    parser_cri = subparsers.add_parser("cri", help="Run the CRI script")

    args = parser.parse_args()

    logger.add("logs/risk-assessment-{time:YYYYMMDD}.log", rotation="00:00")

    if args.action == "encounters":
        timestart = time.time()

        if args.profile:
            run_with_profiling()
        elif args.create_training_data:
            create_training_data()
        else:
            run_main_processing()
        
        timeend = time.time()
        logger.info(f"Total execution time: {timeend - timestart:.2f} seconds")
    elif args.action == "training":
        logger.info("Running VCRA model training...")
        training.run(args.use_checkpoint, args.sample_data)
    elif args.action == "cri":
        start_date = pd.to_datetime(os.getenv("CRI_START_DATE")).date()
        end_date = pd.to_datetime(os.getenv("CRI_END_DATE")).date()

        logger.info("Running CRI calculations for data...")
        vessel_cri.run(DATA_DIR, start_date, end_date)


if __name__ == "__main__":
    main()
