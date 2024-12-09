import os
import numpy as np
import pandas as pd
import traceback

from datetime import datetime, timedelta
from loguru import logger
from utils.cri import calc_cpa, calc_cri

def calc_vessel_cri(data, drop_rows=True, get_cri_values=True, vcra_model=None):
    cri_values = []
    euclidean_distance = []
    rel_movement_direction = []
    azimuth_target_to_own = []

    # Drop rows with missing values
    if drop_rows:
        new_data = data.dropna().copy()
        num_rows_dropped = len(data) - len(new_data)
        if num_rows_dropped > 0:
            logger.warning(f"Dropped {num_rows_dropped} rows with missing values")
            data = new_data

        # Drop rows where the vessel speed and course are identical
        new_data = data.drop(
            data[
                (data["vessel_1_speed"] == data["vessel_2_speed"])
                & (data["vessel_1_course"] == data["vessel_2_course"])
            ].index
        ).copy()

        num_rows_dropped = len(data) - len(new_data)
        if num_rows_dropped > 0:
            logger.warning(
                f"Dropped {num_rows_dropped} rows with identical vessel speed and course"
            )
            data = new_data

    for idx, row in data.iterrows():
        cpa = calc_cpa(row)

        # Extract CPA-related data
        euclidean_distance.append(cpa["euclidian_dist"][0][0])
        rel_movement_direction.append(cpa["rel_movement_direction"])
        azimuth_target_to_own.append(cpa["azimuth_target_to_own"])

        if get_cri_values:
            # Calculate CRI
            if vcra_model:
                vcra_input = np.array(
                    [row['vessel_1_speed'],
                     row['vessel_1_course'],
                     row['vessel_2_speed'],
                     row['vessel_2_course'],
                     cpa['euclidian_dist'][0][0],
                     cpa['azimuth_target_to_own'],
                     cpa['rel_movement_direction']]
                    ).reshape(1, -1)
                cri = max(0, min(1, vcra_model.predict(vcra_input)[0]))
            else:
                cri = calc_cri(
                    row,
                    cpa["euclidian_dist"],
                    cpa["rel_movement_direction"],
                    cpa["azimuth_target_to_own"],
                    cpa["rel_bearing"],
                    cpa["dcpa"],
                    cpa["tcpa"],
                    cpa["rel_speed_mag"],
                )
            cri_values.append(cri)

    # Add the collected data to the DataFrame
    data["euclidian_dist"] = euclidean_distance
    data["rel_movement_direction"] = rel_movement_direction
    data["azimuth_target_to_own"] = azimuth_target_to_own
    
    if get_cri_values:
        data["ves_cri"] = cri_values

    return data


def generate_file_paths(data_dir, start_date, end_date):
    """A generator to yield existing file paths for each date in the range."""
    current_date = start_date
    while current_date <= end_date:
        filename = f"TC_aisdk-{current_date}.csv"
        filepath = os.path.join(f"{data_dir}/training_data_encounters", filename)
        if os.path.exists(filepath):
            yield filepath
        current_date += timedelta(days=1)


def process_and_save_cri(data_dir, start_date, end_date, file_path, tag):
    """Process multiple CSV files, calculate CRI values, and save results to a feather file."""
    results = []  # List to store processed DataFrames

    if file_path:
        file_paths = [file_path]
    else:
        file_paths = generate_file_paths(data_dir, start_date, end_date)

    for filepath in file_paths:
        if os.path.exists(filepath):
            logger.info(f"Processing file: {filepath}")
            try:
                df = pd.read_csv(filepath)
                df = calc_vessel_cri(df, drop_rows=True, get_cri_values=True)
                results.append(df)
            except Exception as e:
                logger.error(
                    f"Error processing {filepath}: {e} {traceback.format_exc()}"
                )
        else:
            logger.warning(f"File not found: {filepath}")

    if results:
        # Concatenate all processed DataFrames and save to a single feather file
        if file_path:
            output_file = os.path.join(
                data_dir, f"training_aisdk_sf{tag}.csv"
            )
        else:
            output_file = os.path.join(
                data_dir, f"training_aisdk_{start_date}_{end_date}{tag}.csv"
            )
        combined_data = pd.concat(results, ignore_index=True)
        combined_data.to_csv(output_file)
        logger.info(f"All data saved to {output_file}")
    else:
        logger.warning("No data to save.")


def run(data_dir, start_date, end_date, file_path=None, tag=""):
    logger.info("Calculating CRI for the training data...")

    process_and_save_cri(data_dir, start_date, end_date, file_path, tag)
