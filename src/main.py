import asyncio
import time
import os
import argparse
import pandas as pd
import encounters.helper as helper
import encounters.vessel_encounter as ve
import server  # TODO: probably change this name
import uvicorn


from concurrent.futures import ProcessPoolExecutor
from cProfile import Profile
from pstats import Stats
from vcra import training, vessel_cri
from loguru import logger
from dotenv import load_dotenv
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.getenv("DATA_DIR", "./data")
AIS_FILE_NAME = os.getenv("AIS_FILE_NAME")
NUMBER_OF_WORKERS = int(os.getenv("NUMBER_OF_WORKERS", 4))

# Server configuration
SOURCE_IP = os.getenv("SOURCE_IP", "0.0.0.0")
SOURCE_PORT = int(os.getenv("SOURCE_PORT", 4571))
TARGET_ENDPOINT_FOR_CURRENT_SHIPS = os.getenv("TARGET_ENDPOINT_FOR_CURRENT_SHIPS")
LOG_LEVEL = os.getenv("SERVER_LOG_LEVEL", "info")
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", 1))


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
    logger.info(
        f"Processing file {current_index} out of {total_files}: {data['file_name']}"
    )
    try:
        if helper.check_if_file_exists(data["file_name"]):
            logger.info(f"File {current_index} already processed: {data['file_name']}")
            return

        file_name_csv = helper.get_AIS_data_file(data["url"])
        ve.vessel_encounters(file_name_csv)
        file_path = os.path.join(DATA_DIR, file_name_csv)
        logger.info(f"Attempting to remove file {file_path}")
        os.remove(file_path)
        logger.info(
            f"Successfully processed and removed file {current_index} out of {total_files}: {data['file_name']}"
        )
    except Exception as e:
        logger.error(
            f"Error processing file {current_index} out of {total_files}: {data['file_name']}, Error: {e}"
        )


def run_with_profiling():
    """Run the main function with profiling."""
    logger.info("Profiling enabled")
    with Profile() as pr:
        run_main_processing()
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        server.app,
        host=SOURCE_IP,
        port=SOURCE_PORT,
        log_level=LOG_LEVEL,
        workers=SERVER_WORKERS,
    )


def main():
    # --------- Single File Risk-Assessment ------------
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
    parser_encounters.add_argument(
        "--create-training-data",
        action="store_true",
        help="Create training data for the ML model",
    )

    # --------- Training data for Risk-Assessment ------------
    parser_training = subparsers.add_parser("training", help="Run the training script")
    # parser_training.add_argument(
    #     "--use-checkpoint",
    #     "-uc",
    #     action="store_true",
    #     help="Flag to use checkpoint for resuming training",
    #     default=False
    # )
    # parser_training.add_argument(
    #     "--sample-data",
    #     "-sd",
    #     type=int,
    #     help="Number of sample data points to generate for training",
    #     default=0
    # )

    # --------- CRI Calculation ------------
    parser_cri = subparsers.add_parser("cri", help="Run the CRI script")
    parser_cri.add_argument(
        "--file-path",
        "-f",
        type=str,
        help="Optional file path for CRI data",
        default=None
    )
    parser_cri.add_argument(
        "--tag",
        "-t",
        type=str,
        help="Optional tag for the CRI data",
        default=''
    )

    # --------- Run cluster + VCRA server ------------
    parser_server = subparsers.add_parser("server", help="Run the server")

    args = parser.parse_args()

    setup_logger()

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
        training.run(DATA_DIR)
    elif args.action == "cri":
        start_date = pd.to_datetime(os.getenv("CRI_START_DATE")).date()
        end_date = pd.to_datetime(os.getenv("CRI_END_DATE")).date()

        logger.info("Running CRI calculations for data...")
        vessel_cri.run(DATA_DIR, start_date, end_date, args.file_path, args.tag)
    elif args.action == "server":
        logger.info("Running the server...")
        run_server()


if __name__ == "__main__":
    main()
