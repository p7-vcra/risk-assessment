import asyncio
import time
import encounters.vessel_encounter as ve
import pandas as pd
import os
import argparse

from cProfile import Profile
from pstats import Stats
from vcra import training, vessel_cri
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def run_encounters():
    src_path = os.getenv('SRC_PATH')
    distance_threshold_in_km = int(os.getenv('DISTANCE_THRESHOLD_IN_KM', 1))
    temporal_threshold_in_seconds = int(os.getenv('TEMPORAL_THRESHOLD_IN_SECONDS', 30))
    time_for_batches_in_s = int(os.getenv('TIME_FOR_BATCHES_IN_S', 12))
    run_from_timestamp = pd.to_datetime(os.getenv('RUN_FROM_TIMESTAMP', "1000-01-01 00:00:00"))
    run_until_timestamp = pd.to_datetime(os.getenv('RUN_UNTIL_TIMESTAMP', "9999-12-31 23:59:59"))

    # Run the data stream
    asyncio.run(
        ve.vessel_encounters(
            src_path,
            distance_threshold_in_km,
            temporal_threshold_in_seconds,
            time_for_batches_in_s,
            run_from_timestamp,
            run_until_timestamp
        )
    )

def main():
    parser = argparse.ArgumentParser(description="Run the risk assessment scripts")
    subparsers = parser.add_subparsers(dest='action', help="Specify the type of script to run")

    parser_encounters = subparsers.add_parser('encounters', help="Run the vessel encounters script")
    parser_encounters.add_argument('--profile', '-p', action='store_true', help="Run with profiling")
    
    parser_training = subparsers.add_parser('training', help="Run the training script")
    parser_training.add_argument('--use-checkpoint', '-uc', action='store_true', help='Flag to use checkpoint for resuming training')
    parser_training.add_argument('--sample-data', '-sd', type=int, help='Number of sample data points to generate for training')

    parser_cri = subparsers.add_parser('cri', help="Run the CRI script")

    args = parser.parse_args()
    
    logger.add("logs/risk-assessment-{time:YYYYMMDD}.log", rotation="00:00")

    if args.action == 'encounters':
        if args.profile:
            logger.info("Running vessel encounters with profiling...")
            with Profile() as pr:
                run_encounters()
            stats = Stats(pr).sort_stats("cumtime")
            stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.
        else:
            logger.info("Running vessel encounters...")
            timestart = time.time()
            run_encounters()
            timeend = time.time()
            print("Time taken: ", timeend - timestart)
    elif args.action == 'training':
        logger.info("Running VCRA model training...")
        training.run(args.use_checkpoint, args.sample_data)
    elif args.action == 'cri':
        logger.info("Running CRI calculations for data...")
        vessel_cri.run("data/training_data_encounters")


if __name__ == "__main__":
    main()
