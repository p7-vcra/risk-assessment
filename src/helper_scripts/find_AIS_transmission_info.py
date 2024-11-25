import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 

def analyze_ais_data(file_path):
    timestart = time.time()
    # Step 1: Load the data into a DataFrame
    #df = pd.read_csv(file_path, parse_dates=['# Timestamp'], dayfirst=True, engine='c')
    df = pd.read_feather(file_path, use_threads=True)

    # Step 2: Convert 'Timestamp' column to datetime format (if not already)
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format='%d/%m/%Y %H:%M:%S')

    # Step 3: Sort the data by MMSI and Timestamp
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    # Step 4: Calculate time differences between consecutive AIS data for each ship (MMSI)
    df['Time Diff'] = df.groupby('MMSI')['# Timestamp'].diff().dt.total_seconds()

    # Step 5: Filter out the first row per MMSI because it will have NaN for 'Time Diff'
    df_filtered = df.dropna(subset=['Time Diff'])

    # Step 6: Calculate statistics for each ship (MMSI)
    stats = df_filtered.groupby('MMSI')['Time Diff'].agg(['min', 'max', 'mean', 'std'])

    # Step 7: Print per-MMSI statistics
    print("Per-MMSI Statistics:")
    print(stats)

    # Step 8: Calculate and print overall statistics
    overall_stats = {
        "mean": df_filtered['Time Diff'].mean(),
        "min": df_filtered['Time Diff'].min(),
        "max": df_filtered['Time Diff'].max(),
        "std": df_filtered['Time Diff'].std()
    }

    print("\nOverall Time Diff Statistics:")
    for stat, value in overall_stats.items():
        print(f"{stat.capitalize()}: {value:.2f} seconds")

    # Step 9: Calculate and print fractiles
    fractiles = df_filtered['Time Diff'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1])
    print("\nTime Diff Fractiles (Percentiles):")
    print(fractiles.to_frame(name="Value (seconds)").rename_axis("Percentile"))

    timeend = time.time()
    print("Time taken: ", timeend - timestart)

# Usage example
#file_path = './data/RealData.csv'  # Update with your actual file path
file_path = './data/RealData1MRows.feather'  # Update with your actual file path
analyze_ais_data(file_path)
