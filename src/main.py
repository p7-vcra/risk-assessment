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

load_dotenv()

PATH_TO_AIS_FILE = os.getenv('SRC_PATH')
NUMBER_OF_WORKERS = int(os.getenv('NUMBER_OF_WORKERS', 4))
SRC_PATH = os.getenv('SRC_PATH')

def main():
    ve.vessel_encounters(PATH_TO_AIS_FILE)

def get_training_data():
    # start with threads 
    AIS_data_info = helper.get_AIS_data_info()
    print("air", AIS_data_info)
    
    # Define a thread pool
    with ProcessPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        # Submit tasks to the executor for each file name
        futures = [
            executor.submit(run_vessel_encounter_for_url, data) # Pass the file name as an argument
            for data in AIS_data_info
        ]
        
        # Wait for all tasks to complete and handle exceptions if any
        for future in asyncio.as_completed(futures):
            try:
                future.result()  # Retrieve the result (or exception)
            except Exception as e:
                print(f"Error during processing: {e}")

def run_vessel_encounter_for_url(data):
    print("Running vessel encounter for date: ", data["file_name"])
    try:
        file_path = helper.get_AIS_data_file(data["url"])
        ve.vessel_encounters(file_path)
        os.remove(file_path)
    except Exception as e:
        print("Error: ", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the risk assessment script with or without profiling.")
    parser.add_argument('--profile', action='store_true', help="Run with profiling")
    parser.add_argument('--create-training-data', action='store_true', help="Create training data for the ML model")
    args = parser.parse_args()
    if args.profile:
        with Profile() as pr:
            main()
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.
    elif args.create_training_data:
        get_training_data()
    else:
        timestart = time.time()
        #main()
        get_training_data()
        timeend = time.time()
        print("Time taken: ", timeend - timestart)