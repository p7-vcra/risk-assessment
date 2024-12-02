import pandas as pd


def find_longest_and_shortest_pair_clustering(save=True):

    # Define the path to your CSV file
    file_path = 'output/TC_aisdk-2024-11-23.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows to verify the data
    print(df.head())

    # Convert the start_time and end_time columns to datetime objects
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Calculate the time duration for each encounter in seconds
    df["duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()

    # Convert duration in seconds to HH:MM:SS format
    df["duration"] = pd.to_datetime(df["duration_seconds"], unit="s").dt.strftime(
        "%H:%M:%S"
    )

    # Sort by distance and duration
    df = df.sort_values(by=["distance", "duration_seconds"])

    # Display the updated DataFrame with duration
    print(df[["vessel_1", "vessel_2", "start_time", "end_time", "duration"]].head())

    # Save the updated DataFrame with the duration column to a new CSV file
    if save:
        df.to_csv("vessel_encounter_data_with_duration.csv", index=False)
    return df

if __name__ == '__main__':
    find_longest_and_shortest_pair_clustering()