import numpy as np
import pandas as pd
import geopandas as gpd
import os
import time
import itertools
import utils.data_util as data_utils
import warnings
import requests
import zipfile
import rarfile

from asyncio import sleep
from datetime import datetime
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from loguru import logger

load_dotenv()

EPS = 1e-9
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "./output")
DATA_DIR = os.getenv("DATA_DIR", "./data")
AIS_DATA_URL = os.getenv("AIS_DATA_URL", "http://web.ais.dk/aisdata/")

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # TODO: Figure out the futurewarnings for concatenation and remove this line


def convert_gdf_from_deg_to_rad(gdf):
    # Use numpy to directly assign converted values for better performance
    radian_cols = ["Latitude", "Longitude", "course"]
    gdf[radian_cols] = np.radians(gdf[radian_cols])
    return gdf


def make_datastream_from_csv(file_name):
    path_to_file = os.path.join(DATA_DIR, file_name)

    # Load only necessary columns and filter data in chunks
    chunk_iter = pd.read_csv(
        path_to_file,
        parse_dates=["# Timestamp"],
        usecols=[
            "# Timestamp",
            "MMSI",
            "Longitude",
            "Latitude",
            "SOG",
            "COG",
            "Length",
        ],
        chunksize=100000,  # Adjust chunk size based on memory capacity
        dayfirst=True,
    )

    for chunk in chunk_iter:
        # Apply speed filter early to minimize processing
        chunk = chunk.loc[(chunk["SOG"] >= 1) & (chunk["SOG"] <= 50)]
        chunk = chunk.drop_duplicates(subset=["# Timestamp", "MMSI"])

        # Create GeoDataFrame and process directly
        chunk["geometry"] = gpd.points_from_xy(chunk["Longitude"], chunk["Latitude"])
        gdf = gpd.GeoDataFrame(
            chunk, crs="EPSG:4326", geometry="geometry"  # Explicit CRS for clarity
        )

        # Rename columns and convert coordinates
        gdf.rename(columns={"SOG": "speed", "COG": "course"}, inplace=True)
        gdf = convert_gdf_from_deg_to_rad(gdf)

        # Stream data by timestamp
        for timestamp, group in gdf.groupby("# Timestamp"):
            yield group


def vessel_encounters(timestamp, new_data, distance_threshold_in_km):
    # a fancy way to check if new_data is empty -> faster
    if new_data.shape[0] == 0:
        return data_utils.create_pair_dataframe()

    # Get current vessel pairs and their distances within the distance threshold
    pairs_current = get_vessel_pairs_in_radius(new_data, distance_threshold_in_km)

    pairs_current = get_MMSI_info_for_current_pairs(timestamp, pairs_current, new_data)

    return pairs_current


def get_vessel_pairs_in_radius(new_data, distance_threshold_in_km):
    timeslice_df = new_data[["Longitude", "Latitude"]].copy()
    timeslice_df.loc[timeslice_df.duplicated()] + EPS

    groups_of_vessels, distances = get_vessels_in_radius(
        timeslice_df, distance_threshold_in_km
    )

    # this will return a list of pairs and their distances.
    # We zip each group_of_vessels with distances (both array of arrays) - group_of_vessles[0] = [0, 1, 2] and distances[0] = [0.0, 1.23, 3.42] respectively (distances in km)
    # We then call get_pairs on a single "group_of_vessel" and its corresponding "distance" array
    pairs = itertools.chain.from_iterable(
        [
            get_pairs(groups_of_vessels, distances)
            for groups_of_vessels, distances in zip(groups_of_vessels, distances)
        ]
    )

    return pairs


def get_pairs(group_of_vessel, distances):
    # Optimize with list comprehension for better performance
    return [
        (group_of_vessel[0], other_vessel, distance)
        for other_vessel, distance in zip(group_of_vessel[1:], distances[1:])
    ]


def get_vessels_in_radius(new_data, distance_threshold_in_km=1.852):

    new_data_long_lat = new_data[["Latitude", "Longitude"]]

    df_tree = BallTree(new_data_long_lat, leaf_size=40, metric="haversine")

    distance_threshold_in_m = distance_threshold_in_km * 1000

    diam = distance_threshold_in_m / (6371 * 1000)

    # Find all pairs of vessels within the distance threshold
    points, distance = df_tree.query_radius(
        new_data_long_lat, r=diam, return_distance=True, sort_results=True
    )  # 6371.0088 is the radius of the Earth in km

    mask = np.vectorize(lambda l: len(l) > 1)(points)
    return points[mask], distance[mask] * 6371


def get_MMSI_info_for_current_pairs(timestamp, pairs, new_data):
    # Convert `new_data` to a dictionary for quick lookup
    vessel_data = new_data.to_dict("index")

    # Prepare lists for the final DataFrame columns
    data_records = []
    seen_pairs = set()

    # Iterate over the pairs and build the records list
    for vessel_1, vessel_2, distance in pairs:
        mmsi_1 = vessel_data[vessel_1]["MMSI"]
        mmsi_2 = vessel_data[vessel_2]["MMSI"]

        if mmsi_1 != mmsi_2:
            # Sort MMSIs to maintain consistent pair order
            sorted_pair = tuple(sorted((mmsi_1, mmsi_2)))

            # Skip if this pair has already been processed
            if sorted_pair not in seen_pairs:
                # Fetch vessel data once
                data_1 = vessel_data[vessel_1]
                data_2 = vessel_data[vessel_2]

                # Add the record for this pair
                data_records.append(
                    {
                        "vessel_1": sorted_pair[0],
                        "vessel_2": sorted_pair[1],
                        "vessel_1_longitude": data_1["Longitude"],
                        "vessel_2_longitude": data_2["Longitude"],
                        "vessel_1_latitude": data_1["Latitude"],
                        "vessel_2_latitude": data_2["Latitude"],
                        "vessel_1_speed": data_1["speed"],
                        "vessel_2_speed": data_2["speed"],
                        "vessel_1_course": data_1["course"],
                        "vessel_2_course": data_2["course"],
                        "vessel_1_length": data_1["Length"],
                        "vessel_2_length": data_2["Length"],
                        "distance": distance,
                        "start_time": timestamp,
                        "end_time": timestamp,
                    }
                )

                # Mark the pair as seen
                seen_pairs.add(sorted_pair)

    # If no valid pairs were processed, return an empty DataFrame
    if not data_records:
        return data_utils.create_pair_dataframe()

    # Create a DataFrame from the collected records
    df_pairs = pd.DataFrame.from_records(data_records)

    # Set a MultiIndex for easier querying
    df_pairs.set_index(["vessel_1", "vessel_2"], inplace=True)

    # Sanity check to ensure no duplicates exist
    assert not df_pairs.index.duplicated().any(), "Duplicate pairs found"

    return df_pairs


def temp_output_to_file(file_name, pairs_out):
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    pairs_out = pairs_out.reset_index()
    pairs_out = pairs_out.drop_duplicates(
        subset=["vessel_1", "vessel_2", "start_time"], keep="last"
    )
    pairs_out = pairs_out.set_index(
        ["vessel_1", "vessel_2"]
    )  # Restore the original index structure if needed

    # write to output file that
    current_time = time.strftime("%Y%m%d-%H:%M:%S")
    logger.info("Writing to output file at: ", current_time)
    pairs_out.index.names = ["vessel_1", "vessel_2"]
    pairs_out.to_csv(f"./output/TC_{file_name}", mode="w", header=True, index=True)


def update_pairs(timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds):
    # Identify disappeared and emerged pairs using set operations
    disappeared_df = active_pairs.loc[~active_pairs.index.isin(current_pairs.index)]
    emerged_df = current_pairs.loc[~current_pairs.index.isin(active_pairs.index)]

    # Merge active_pairs and current_pairs on indices to identify common pairs
    common_pairs = active_pairs.merge(
        current_pairs,
        how="inner",
        left_index=True,
        right_index=True,
        suffixes=("_active", "_current"),
    )

    # Identify survived and disappeared pairs among common indices
    survived_mask = common_pairs["distance_current"] <= common_pairs["distance_active"]
    disappeared_mask = ~survived_mask

    # Survived pairs: Update start_time from active_pairs and set end_time to the current timestamp
    survived_df = current_pairs.loc[common_pairs.loc[survived_mask].index]
    survived_df["start_time"] = common_pairs.loc[survived_mask, "start_time_active"]
    survived_df["end_time"] = timestamp

    # Disappeared pairs: From active_pairs but did not survive
    disappeared_pairs = active_pairs.loc[common_pairs.loc[disappeared_mask].index]
    disappeared_df = pd.concat([disappeared_df, disappeared_pairs])

    # Filter disappeared pairs for those exceeding the temporal threshold
    disappeared_df = disappeared_df[
        (disappeared_df["end_time"] - disappeared_df["start_time"]).dt.total_seconds()
        >= temporal_threshold_in_seconds
    ]

    # Update active pairs with survived and emerged pairs
    updated_active_pairs = pd.concat([emerged_df, survived_df])

    # Update the end_time of all active pairs to the current timestamp
    updated_active_pairs["end_time"] = timestamp

    # Return updated active and inactive pairs
    return updated_active_pairs, disappeared_df


def get_AIS_data_info():
    try:
        logger.info(f"Downloading AIS data from {AIS_DATA_URL}")
        response = requests.get(AIS_DATA_URL)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")[4:]  # Skip the first 4 rows
        data = []
        for row in rows[:-1]:  # Skip the last row
            cols = row.find_all("td")
            if len(cols) >= 4:
                entry = {
                    "file_name": cols[1].text.strip(),
                    "url": AIS_DATA_URL + cols[1].text.strip(),
                    "size": cols[3].text.strip(),
                }
            data.append(entry)
        data.reverse()
        return data

    except Exception as e:
        logger.error(
            f"Failed to download AIS data. Please check the URL and try again. Error: {e}"
        )
        return


def get_AIS_data_file(url_name):
    try:
        # if src folder doestn exist create it
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        logger.info(f"Downloading AIS data file: {url_name}")
        response = requests.get(url_name, stream=True)
        response.raise_for_status()

        # Get file name from URL
        file_name = url_name.split("/")[-1]
        file_path = os.path.join(DATA_DIR, file_name)
        file_path_csv = file_path.replace(".zip", ".csv").replace(".rar", ".csv")
        file_name_csv = file_name.replace(".zip", ".csv").replace(".rar", ".csv")

        if os.path.exists(file_path_csv):
            logger.info(f"File already exists: {file_path_csv}")
            return file_name_csv

        # Get the total file size from the headers
        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

        # Save the downloaded file with progress
        with open(file_path, "wb") as f:
            downloaded_size = 0
            percent_complete = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    new_percent_complete = int((downloaded_size / total_size) * 100)
                    if new_percent_complete >= percent_complete + 10:
                        percent_complete = new_percent_complete
                        logger.info(
                            f"Download progress: {percent_complete}% for {file_name}"
                        )

        logger.info(f"File downloaded: {file_path}")

        # Check and extract the file
        if zipfile.is_zipfile(file_path):
            logger.info("Extracting ZIP file... ")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            logger.info("ZIP file extracted successfully.")
        elif rarfile.is_rarfile(file_path):
            logger.info("Extracting RAR file...")
            with rarfile.RarFile(file_path, "r") as rar_ref:
                rar_ref.extractall(DATA_DIR)
            logger.info("RAR file extracted successfully.")
        else:
            logger.warning("The downloaded file is neither a ZIP nor a RAR file.")

        # Optionally delete the archive after extraction
        os.remove(file_path)
        logger.info(f"Archive file {file_name} has been removed.")
        return file_name_csv

    except Exception as e:
        logger.error(f"Failed to download or extract AIS data file. Error: {e}")


def check_if_file_exists(file_name):
    file_path_csv = os.path.join(
        OUTPUT_DIRECTORY,
        "TC_" + file_name.replace(".zip", ".csv").replace(".rar", ".csv"),
    )
    return os.path.exists(file_path_csv)
