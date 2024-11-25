import pandas as pd

def filter_mmsi_data(file_path, mmsi_1, mmsi_2, start_timestamp):
    df = pd.read_csv(file_path)  # Handle potential comment rows
    
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format='%m/%d/%Y %H:%M:%S')
    
    start_time = pd.to_datetime(start_timestamp, format='%m/%d/%Y %H:%M:%S')
    
    filtered_df = df[
        ((df['MMSI'] == int(mmsi_1)) | (df['MMSI'] == int(mmsi_2))) & 
        (df['# Timestamp'] >= start_time)
    ]
    
    filtered_df = filtered_df[['# Timestamp', 'MMSI', 'Latitude', 'Longitude']]  # Keep only relevant columns

    return filtered_df

file_path = "./data/testingCurrently.csv"

mmsi_1 = "277550000"  # Example MMSI
mmsi_2 = "354537000"  # Example MMSI
start_timestamp = "10/09/2024 00:03:00"  # Example start timestamp

result_df = filter_mmsi_data(file_path, mmsi_1, mmsi_2, start_timestamp)

if not result_df.empty:
    print("Finished! Resulting rows:", len(result_df))
    
    # Sort the dataframe by the timestamp
    result_df = result_df.sort_values(by='# Timestamp')
    
    # Open the file to write
    with open("filtered_mmsi_data.csv", "w") as file:
        # Write the header
        file.write("# Timestamp,MMSI,Latitude,Longitude\n")
        
        # Group by # Timestamp and write each group
        for timestamp, group in result_df.groupby('# Timestamp'):
            group.to_csv(file, index=False, header=False)
            file.write("\n")  # Add a newline after each timestamp groupe

