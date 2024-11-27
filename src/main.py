from cProfile import Profile
from pstats import Stats
from dotenv import load_dotenv
import asyncio
import time
import encounters.vessel_encounter as ve
import pandas as pd
import os
import argparse

load_dotenv()

SRC_PATH = os.getenv('SRC_PATH')
DISTANCE_THRESHOLD_IN_KM = int(os.getenv('DISTANCE_THRESHOLD_IN_KM', 1))
TEMPORAL_THRESHOLD_IN_SECONDS = int(os.getenv('TEMPORAL_THRESHOLD_IN_SECONDS', 30))
TIME_FOR_BATCHES_IN_S = int(os.getenv('TIME_FOR_BATCHES_IN_S', 12))
RUN_UNTIL_TIMESTAMP = pd.to_datetime(os.getenv('RUN_UNTIL_TIMESTAMP', "9999-12-31 23:59:59"))
RUN_FROM_TIMESTAMP = pd.to_datetime(os.getenv('RUN_FROM_TIMESTAMP', "1000-01-01 00:00:00"))

def main():
    # Run the data stream
    asyncio.run(ve.vessel_encounters(SRC_PATH, DISTANCE_THRESHOLD_IN_KM, TEMPORAL_THRESHOLD_IN_SECONDS, TIME_FOR_BATCHES_IN_S, RUN_FROM_TIMESTAMP, RUN_UNTIL_TIMESTAMP))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the risk assessment script with or without profiling.")
    parser.add_argument('--profile', action='store_true', help="Run with profiling")
    args = parser.parse_args()
    if args.profile:
        with Profile() as pr:
            main()
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.
    else:
        timestart = time.time()
        main()
        timeend = time.time()
        print("Time taken: ", timeend - timestart)