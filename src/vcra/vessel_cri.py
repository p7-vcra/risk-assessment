import os
import pandas as pd
import traceback

from datetime import datetime, timedelta
from loguru import logger
from utils.cri import calc_cpa, calc_cri


def calc_vessel_cri(data):
    cri_values = []
    euclidean_distance = []
    rel_movement_direction = []
    azimuth_target_to_own = []

    # Drop rows with missing values
    new_data = data.dropna()
    logger.info(f"Dropped {len(data) - len(new_data)} rows with missing values")
    data = new_data

    # Drop rows where the vessel speed and course are identical
    new_data = data.drop(
        data[
            (data["vessel_1_speed"] == data["vessel_2_speed"])
            & (data["vessel_1_course"] == data["vessel_2_course"])
        ].index
    )
    logger.info(
        f"Dropped {len(data) - len(new_data)} rows with identical vessel speed and course"
    )
    data = new_data

    for idx, row in data.iterrows():
        cpa = calc_cpa(row)

        # Extract CPA-related data
        euclidean_distance.append(cpa["euclidian_dist"][0][0])
        rel_movement_direction.append(cpa["rel_movement_direction"])
        azimuth_target_to_own.append(cpa["azimuth_target_to_own"])

        # Calculate CRI
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
    data["euclidean_distance"] = euclidean_distance
    data["rel_movement_direction"] = rel_movement_direction
    data["azimuth_target_to_own"] = azimuth_target_to_own
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


def process_and_save_cri(data_dir, start_date, end_date):
    """Process multiple CSV files, calculate CRI values, and save results to a feather file."""
    results = []  # List to store processed DataFrames

    for filepath in generate_file_paths(data_dir, start_date, end_date):
        if os.path.exists(filepath):
            logger.info(f"Processing file: {filepath}")
            try:
                df = pd.read_csv(filepath)
                df = calc_vessel_cri(df)
                results.append(df)
            except Exception as e:
                logger.error(
                    f"Error processing {filepath}: {e} {traceback.format_exc()}"
                )
        else:
            logger.warning(f"File not found: {filepath}")

    if results:
        # Concatenate all processed DataFrames and save to a single feather file
        output_file = os.path.join(
            data_dir, f"training_aisdk_{start_date}_{end_date}.csv"
        )
        combined_data = pd.concat(results, ignore_index=True)
        combined_data.to_csv(output_file)
        logger.info(f"All data saved to {output_file}")
    else:
        logger.warning("No data to save.")


def run(data_dir, start_date, end_date):
    logger.info("Calculating CRI for the training data...")

    process_and_save_cri(data_dir, start_date, end_date)
