import pandas as pd

pairType = {
    "vessel_1_MMSI": "int64",
    "vessel_2_MMSI": "int64",
    "distance": "float64",
    "start_time": "datetime64[ns]",
    "end_time": "datetime64[ns]",
    "vessel_1_longitude": "float64",
    "vessel_2_longitude": "float64",
    "vessel_1_latitude": "float64",
    "vessel_2_latitude": "float64",
    "vessel_1_speed": "float64",
    "vessel_2_speed": "float64",
    "vessel_1_course": "float64",
    "vessel_2_course": "float64",
    "vessel_1_length": "float64",
    "vessel_2_length": "float64",
}


def create_pair_dataframe():
    columns = [
        "vessel_1_MMSI",
        "vessel_2_MMSI",
        "vessel_1",
        "vessel_2",
        "distance",
        "start_time",
        "end_time",
        "vessel_1_longitude",
        "vessel_2_longitude",
        "vessel_1_latitude",
        "vessel_2_latitude",
        "vessel_1_speed",
        "vessel_2_speed",
        "vessel_1_course",
        "vessel_2_course",
        "vessel_1_length",
        "vessel_2_length",
    ]
    # Create the DataFrame
    df = pd.DataFrame(columns=columns)  # Ensure the correct data types
    # Set the index to be MultiIndex with vessel_1 and vessel_2
    df.set_index(["vessel_1", "vessel_2"], inplace=True)
    return df
