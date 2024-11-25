import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_ais_data(file_path):
    # Step 1: Load the data into a DataFrame
    df = pd.read_csv(file_path, parse_dates=['# Timestamp'], dayfirst=True)

    # Step 2: Convert 'Timestamp' column to datetime format (if not already)
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format='%d/%m/%Y %H:%M:%S')

    # Step 3: Sort the data by MMSI and Timestamp
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    # Step 4: Calculate time differences between consecutive AIS data for each ship (MMSI)
    df['Time Diff'] = df.groupby('MMSI')['# Timestamp'].diff().dt.total_seconds()

    # Step 5: Filter out the first row per MMSI because it will have NaN for 'Time Diff'
    df_filtered = df.dropna(subset=['Time Diff'])

    # **Cut off values above 900 seconds**
    df_filtered = df_filtered[df_filtered['Time Diff'] <= 100]

    # Step 6: Calculate statistics for each ship (MMSI)
    stats = df_filtered.groupby('MMSI')['Time Diff'].agg(['min', 'max', 'mean', 'std'])

    # Step 7: Display statistics
    print("Statistics for AIS data transmission intervals by ship (MMSI):")
    print(stats)

    # Step 8: Visualize the distribution of time intervals
    plt.figure(figsize=(20, 6))
    sns.histplot(df_filtered['Time Diff'], kde=True, bins=30, color='blue', stat='density')
    plt.title('Distribution of Time Intervals Between AIS Transmissions')
    plt.xlabel('Time Interval (seconds)')
    plt.ylabel('Density')
    plt.show()

# Usage example
file_path = './data/RealData1MRows.csv'  # Update with your actual file path
analyze_ais_data(file_path)
