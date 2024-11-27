import time
import asyncio
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import encounters.helper as helper
import utils.data_util as du
from dotenv import load_dotenv

load_dotenv()

DISTANCE_THRESHOLD_IN_KM = int(os.getenv('DISTANCE_THRESHOLD_IN_KM', 1))
TEMPORAL_THRESHOLD_IN_SECONDS = int(os.getenv('TEMPORAL_THRESHOLD_IN_SECONDS', 30))
TIME_FOR_BATCHES_IN_S = int(os.getenv('TIME_FOR_BATCHES_IN_S', 12))
RUN_UNTIL_TIMESTAMP = pd.to_datetime(os.getenv('RUN_UNTIL_TIMESTAMP'))
RUN_FROM_TIMESTAMP = pd.to_datetime(os.getenv('RUN_FROM_TIMESTAMP'))


def vessel_encounters(file_name):
    active_pairs = du.create_pair_dataframe()
    inactive_pairs = du.create_pair_dataframe()
    all_outputs = du.create_pair_dataframe()
    
    last_logged_minute = None
    batch_data = []  # Buffer to store data for the batch
    last_batch_time = RUN_FROM_TIMESTAMP

    # Stream data asynchronously from CSV
    for data in helper.make_datastream_from_csv(file_name):
        timestamp = data['# Timestamp'].iloc[0]
        if timestamp < RUN_FROM_TIMESTAMP:
            continue
        
        # Add data to batch
        batch_data.append(data)

        # Check if it's time to process the batch
        if (timestamp - last_batch_time).total_seconds() >= TIME_FOR_BATCHES_IN_S:
            # Concatenate all batch data for processing
            batch_df = pd.concat(batch_data, ignore_index=True)
            batch_df = batch_df.drop_duplicates(subset='MMSI', keep='last', ignore_index=True)

            # Rename all timestamps in the batch to the start time of the batch
            batch_df['# Timestamp'] = last_batch_time

            batch_data = []  # Reset buffer
            last_batch_time = timestamp

            # Process the batch data
            pairs_out = du.create_pair_dataframe()
            current_pairs = helper.vessel_encounters(timestamp, batch_df, DISTANCE_THRESHOLD_IN_KM)
            current_minute = timestamp.floor('min')  # Floor to the nearest minute

            if active_pairs.empty:  # Case 1: active_pairs is empty
                active_pairs = current_pairs
            elif current_pairs.empty:  # Case 2: active pairs but no current pairs
                pairs_out = active_pairs[active_pairs['end_time'] - active_pairs['start_time'] >= pd.Timedelta(seconds=TEMPORAL_THRESHOLD_IN_SECONDS)]
                inactive_pairs = pd.concat([inactive_pairs, pairs_out])
                active_pairs = du.create_pair_dataframe()
            else:  # Both active_pairs and current_pairs are not empty
                active_pairs, new_inactive_pairs = helper.update_pairs(timestamp, active_pairs, current_pairs, TEMPORAL_THRESHOLD_IN_SECONDS)
                inactive_pairs = pd.concat([inactive_pairs, new_inactive_pairs])
                pairs_out = active_pairs[active_pairs['end_time'] - active_pairs['start_time'] >= pd.Timedelta(seconds=TEMPORAL_THRESHOLD_IN_SECONDS)]

            if not pairs_out.empty:
                all_outputs = pd.concat([all_outputs, pairs_out])
            # Print progress message when a new minute starts
            if last_logged_minute is None or current_minute > last_logged_minute:
                print(f"Current timestamp: {timestamp}")
                last_logged_minute = current_minute

        # Exit condition for debugging
        if timestamp == RUN_UNTIL_TIMESTAMP:
            break

    # Final processing if there is leftover data in the batch
    if batch_data:
        batch_df = pd.concat(batch_data, ignore_index=True)
        # Rename all timestamps in the final batch to the last batch time
        batch_df['# Timestamp'] = last_batch_time
        # Process remaining batch logic here if needed

    helper.temp_output_to_file(PATH_TO_AIS_FILE, all_outputs)
