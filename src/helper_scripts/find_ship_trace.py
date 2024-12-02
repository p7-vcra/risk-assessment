import pandas as pd


def filter_mmsi_data(src_path, mmsis, cluster_df):
    # Load the real data
    df = pd.read_csv(src_path)

    # Ensure timestamps are parsed correctly
    df["# Timestamp"] = pd.to_datetime(df["# Timestamp"], format="%m/%d/%Y %H:%M:%S")

    # Create a dictionary of MMSI to minimum start time from the cluster data
    min_start_times = (
        cluster_df.melt(
            id_vars=["distance", "start_time", "end_time"],
            value_vars=["vessel_1", "vessel_2"],
            var_name="vessel_role",
            value_name="MMSI",
        )
        .groupby("MMSI")["start_time"]
        .min()
    )

    # Convert start times to datetime for filtering
    min_start_times = pd.to_datetime(min_start_times)

    # Filter the dataframe for relevant MMSIs and timestamps greater than or equal to the minimum start time
    filtered_rows = []
    for mmsi in mmsis:
        if mmsi in min_start_times:
            start_time = min_start_times[mmsi]
            mmsi_filtered = df[
                (df["MMSI"] == int(mmsi)) & (df["# Timestamp"] >= start_time)
            ]
            filtered_rows.append(mmsi_filtered)

    # Concatenate all filtered rows into a single DataFrame
    filtered_df = (
        pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()
    )

    # Keep only relevant columns
    filtered_df = filtered_df[["# Timestamp", "MMSI", "Latitude", "Longitude"]]

    return filtered_df


def main():
    # Example usage
    file_path = "./data/RealData.csv"
    cluster_path = "./output/output_20241125-12:38:30_cutdown.csv"

    # Load the cluster data
    cluster_df = pd.read_csv(cluster_path)

    # Get unique MMSIs from cluster data
    mmsis = pd.concat([cluster_df["vessel_1"], cluster_df["vessel_2"]]).unique()

    # Filter the data
    filtered_df = filter_mmsi_data(file_path, mmsis, cluster_df)

    # Save the filtered data to a file
    filtered_df.to_csv("filtered_mmsi_data.csv", index=False)
    if not filtered_df.empty:
        print("Finished! Resulting rows:", len(filtered_df))
    else:
        print("No matching data found.")


if __name__ == "__main__":
    main()
