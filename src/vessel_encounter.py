
import time
import asyncio 
import numpy as np
import pandas as pd
import geopandas as gpd
import src.helper as helper
import src.data_util as du

RUN_UNTIL_TIMESTAMP = pd.to_datetime("2024-10-09 00:05:00")
RUN_FROM_TIMESTAMP = pd.to_datetime("2024-10-09 00:00:00")

async def vessel_encounters(SRC_PATH, DISTANCE_THRESHOLD_IN_KM, TEMPORAL_THRESHOLD_IN_SECONDS, TIME_BETWEEN_EACH_DATA_SENT_IN_S):

    active_pairs = du.create_pair_dataframe()
    inactive_pairs = du.create_pair_dataframe()
    all_outputs = du.create_pair_dataframe()

    last_logged_minute = None

    # Stream data asynchronously from CSV
    async for data in helper.make_datastream_from_csv(SRC_PATH, s=TIME_BETWEEN_EACH_DATA_SENT_IN_S):
        timestamp = data['# Timestamp'].iloc[0]
        if timestamp < RUN_FROM_TIMESTAMP:
            continue

        pairs_out = du.create_pair_dataframe()
        current_pairs = helper.vessel_encounters(timestamp, data, DISTANCE_THRESHOLD_IN_KM)
        current_minute = timestamp.floor('min')  # Floor to the nearest minute

        if active_pairs.empty: # Case 1: active_pairs is empty
            active_pairs = current_pairs
        elif current_pairs.empty: # Case 2: we have active pairs but no current pairs
            #set variable pairs_out to be all pairs in active_pairs that have time difference greater than TEMPORAL_THRESHOLD_IN_SECONDS
            pairs_out = active_pairs[active_pairs['end_time'] - active_pairs['start_time'] >= pd.Timedelta(seconds=TEMPORAL_THRESHOLD_IN_SECONDS)] # output pairs that are active
            inactive_pairs = pd.concat([inactive_pairs, pairs_out]) 
            active_pairs = du.create_pair_dataframe()
        else: # Both active_pairs and current_pairs are not empty
            active_pairs, new_inactive_pairs = helper.update_pairs(timestamp, active_pairs, current_pairs, TEMPORAL_THRESHOLD_IN_SECONDS)
            inactive_pairs = pd.concat([inactive_pairs, new_inactive_pairs])
            pairs_out = active_pairs[active_pairs['end_time'] - active_pairs['start_time'] >= pd.Timedelta(seconds=TEMPORAL_THRESHOLD_IN_SECONDS)]  
        if not pairs_out.empty:
            all_outputs = pd.concat([all_outputs, pairs_out])
        #----------------------HERE WE WOULD CALL VESSEL RISK ASSESSMENT MODEL----------------------
        
        # Print progress message when a new minute starts
        if last_logged_minute is None or current_minute > last_logged_minute:
            print(f"Current timestamp: {timestamp}")
            last_logged_minute = current_minute

        #TODO: temp for debugging
        if timestamp == RUN_UNTIL_TIMESTAMP:
            break
        await asyncio.sleep(TIME_BETWEEN_EACH_DATA_SENT_IN_S)

    helper.temp_output_to_file(all_outputs)
    return all_outputs