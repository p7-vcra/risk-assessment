import numpy as np
import pandas as pd
import geopandas as gpd
import os
import time 
import itertools
import utils.data_util as data_utils
import warnings
from asyncio import sleep
from datetime import datetime
from geopy.distance import geodesic
from sklearn.neighbors import BallTree

EPS = 1e-9
MAX_NUMBER_OF_OUTPUT_FILES = 5
assert MAX_NUMBER_OF_OUTPUT_FILES > 0, "MAX_NUMBER_OF_OUTPUT_FILES must be greater than 0"

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=FutureWarning) # TODO: Figure out the futurewarnings for concatenation and remove this line

def convert_gdf_from_deg_to_rad(gdf):
    gdf[['Latitude', 'Longitude', 'course']] = np.deg2rad(gdf[['Latitude', 'Longitude', 'course']])
    return gdf


async def make_datastream_from_csv(path_to_file, s = 1):
    '''Temporary function until server is setup to feed data'''

    #TODO: Fix this pleaseeeeeee-> Converts twice-> no need to
    df = pd.read_csv(path_to_file, parse_dates=['# Timestamp'])

    gdf = gpd.GeoDataFrame(
        df, 
        crs=4326, # this is the crs for lat/lon
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])
    )

    gdf.rename({'SOG':'speed', 'COG':'course'}, axis=1, inplace=True)

    # Drop rows with speed under 1 and speed over 50

    gdf = gdf.loc[gdf['speed'] >= 1 & (gdf['speed'] <= 50)]

    gdf.reset_index(drop=True, inplace=True)

    gdf = convert_gdf_from_deg_to_rad(gdf)

    # while there are still timestamps to iterate over
    for timestamp, group in gdf.groupby('# Timestamp'):
        yield group
        await sleep(s)
    ("Finished streaming data")


def vessel_encounters(timestamp, new_data, distance_threshold_in_km):
    if new_data.empty:
        return data_utils.create_pair_dataframe()

    # Get current vessel pairs and their distances within the distance threshold
    pairs_current = get_vessel_pairs_in_radius(new_data, distance_threshold_in_km)

    pairs_current = get_MMSI_info_for_current_pairs(timestamp, pairs_current, new_data)

    return pairs_current

def get_vessel_pairs_in_radius(new_data, distance_threshold_in_km):
    timeslice_df = new_data[['Longitude', 'Latitude']].copy()
    timeslice_df.loc[timeslice_df.duplicated()] + EPS

    groups_of_vessels, distances = get_vessels_in_radius(timeslice_df, distance_threshold_in_km)

    # this will return a list of pairs and their distances. 
    # We zip each group_of_vessels with distances (both array of arrays) - group_of_vessles[0] = [0, 1, 2] and distances[0] = [0.0, 1.23, 3.42] respectively (distances in km)
    # We then call get_pairs on a single "group_of_vessel" and its corresponding "distance" array
    pairs = itertools.chain.from_iterable([get_pairs(groups_of_vessels, distances) for groups_of_vessels, distances in zip(groups_of_vessels, distances)])

    return pairs


def get_pairs(group_of_vessel, distances):
    return list(zip(itertools.repeat(group_of_vessel[0]), group_of_vessel[1:], distances[1:])) # makes a list of pairs and their distances -> will look like [(0, 1, 1.23), (0, 2, 3.42),] 


def get_vessels_in_radius(new_data, distance_threshold_in_km = 1.852):

    new_data_long_lat = new_data[['Latitude', 'Longitude']]

    df_tree = BallTree(new_data_long_lat, leaf_size=40, metric='haversine')

    distance_threshold_in_m = distance_threshold_in_km * 1000

    diam = distance_threshold_in_m / (6371 * 1000)

    # Find all pairs of vessels within the distance threshold
    points, distance = df_tree.query_radius(new_data_long_lat, r = diam, return_distance = True, sort_results = True) # 6371.0088 is the radius of the Earth in km

    mask = np.vectorize(lambda l: len(l) > 1)(points)
    return points[mask], distance[mask] * 6371

# TODO - TEMP-> THIS ONE REMOVES THE DUPLICATE PAIRS (where MMSI is just flipped) - BUT WE MIGHT NEED THIS
def get_MMSI_info_for_current_pairs(timestamp, pairs, new_data):
    pairs_list = []
    seen_pairs = set()

    # Cache the 'new_data' for efficiency
    vessel_data = new_data.to_dict('index')  # Convert to dictionary of row data for quicker lookup

    for vessel_1, vessel_2, distance in pairs:
        mmsi_1 = vessel_data[vessel_1]['MMSI']
        mmsi_2 = vessel_data[vessel_2]['MMSI']

        if mmsi_1 != mmsi_2:
            # Ensure the pair is always stored in a sorted order
            sorted_pair = tuple(sorted((mmsi_1, mmsi_2)))

            # Check if the pair has already been added
            if sorted_pair not in seen_pairs:
                # Extract relevant data once
                data_1 = vessel_data[vessel_1]
                data_2 = vessel_data[vessel_2]

                # Add the pair to the list
                pairs_list.append({
                    'vessel_1': sorted_pair[0],
                    'vessel_2': sorted_pair[1],
                    'vessel_1_longitude': data_1['Longitude'],  
                    'vessel_2_longitude': data_2['Longitude'],
                    'vessel_1_latitude': data_1['Latitude'],
                    'vessel_2_latitude': data_2['Latitude'],
                    'vessel_1_speed': data_1['speed'],
                    'vessel_2_speed': data_2['speed'],
                    'vessel_1_course': data_1['course'],
                    'vessel_2_course': data_2['course'],
                    'vessel_1_length': data_1['Length'],
                    'vessel_2_length': data_2['Length'],
                    'distance': distance,
                    'start_time': timestamp,
                    'end_time': timestamp
                })
                seen_pairs.add(sorted_pair)

    if not pairs_list:
        return data_utils.create_pair_dataframe()

    # Create the dataframe in one go
    df_pairs = pd.DataFrame.from_records(pairs_list)

    # Create a multi-index from vessel_1 and vessel_2
    df_pairs.set_index(['vessel_1', 'vessel_2'], inplace=True)

    # Ensure there are no duplicates
    df_pairs = df_pairs[~df_pairs.index.duplicated(keep='first')]

    assert not df_pairs.index.duplicated().any(), "Duplicate pairs found"
    
    return df_pairs


def temp_output_to_file(pairs_out):
    output_directory = "./output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #check if more than 5 files in output directory
    files = os.listdir(output_directory)
    if len(files) > MAX_NUMBER_OF_OUTPUT_FILES:
        #remove the oldest files until there are only 5 files
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_directory, x)))
        for i in range(len(files) - MAX_NUMBER_OF_OUTPUT_FILES + 1):
            os.remove(os.path.join(output_directory, files[i]))

    #write to output file that 
    current_time = time.strftime("%Y%m%d-%H:%M:%S")
    print("Writing to output file at: ", current_time)
    pairs_out.index.names = ['vessel_1', 'vessel_2']
    pairs_out.to_csv(f"./output/output_{current_time}.txt", mode='w', header=True, index=True)

def update_pairs(timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds):
    # Initialize DataFrames for survived, disappeared, and emerged pairs
    survived = []
    disappeared = []
    
    # Identify disappeared and emerged pairs
    disappeared_df = active_pairs.loc[~active_pairs.index.isin(current_pairs.index)]
    emerged_df = current_pairs.loc[~current_pairs.index.isin(active_pairs.index)]

    # Filter active_pairs to only those also in current_pairs
    common_indices = active_pairs.index.intersection(current_pairs.index)
    active_common = active_pairs.loc[common_indices].copy()
    current_common = current_pairs.loc[common_indices].copy()

    # Iterate over common indices to update distances and times
    for pair_idx in common_indices:
        pair_active = active_common.loc[pair_idx]
        pair_current = current_common.loc[pair_idx]

        # If active pair's distance is less than current pair's, mark as disappeared
        if pair_active['distance'] < pair_current['distance']:
            disappeared.append(pair_active)
        else:
            # Update start_time directly in the current_pairs DataFrame
            current_pairs.at[pair_idx, 'start_time'] = pair_active['start_time']
            survived.append(current_pairs.loc[pair_idx])

    # Concatenate results for survived and disappeared pairs
    survived_df = pd.DataFrame(survived, index=[pair.name for pair in survived]) if survived else data_utils.create_pair_dataframe()
    disappeared_df = pd.concat([disappeared_df, pd.DataFrame(disappeared)]) if disappeared else disappeared_df

    # Ensure correct data types
    survived_df = survived_df.astype(data_utils.pairType)
    disappeared_df = disappeared_df.astype(data_utils.pairType)

    # Combine emerged and survived pairs to update active_pairs
    updated_active_pairs = pd.concat([emerged_df, survived_df])

    # Filter disappeared pairs to include only those exceeding the time threshold
    threshold_exceeded = disappeared_df[
        (disappeared_df['end_time'] - disappeared_df['start_time']).dt.total_seconds() >= temporal_threshold_in_seconds
    ]
    updated_inactive_pairs = threshold_exceeded

    # Update the end_time of all active pairs to the current timestamp
    updated_active_pairs['end_time'] = timestamp

    # Ensure correct data types
    updated_active_pairs = updated_active_pairs.astype(data_utils.pairType)
    updated_inactive_pairs = updated_inactive_pairs.astype(data_utils.pairType)

    # Return updated active and inactive pairs
    return updated_active_pairs, updated_inactive_pairs

