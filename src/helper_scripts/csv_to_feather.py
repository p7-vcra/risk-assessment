import pandas as pd

# Function to convert CSV to Feather
def csv_to_feather(csv_file_path, feather_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, parse_dates=['# Timestamp'], dayfirst=True, engine='c')
    
    # Save the DataFrame as a Feather file
    df.to_feather(feather_file_path)

# Example usage
csv_file_path = './data/RealData1MRows.csv'  # Replace with your CSV file path
feather_file_path = './data/RealData1MRows.feather'  # Replace with your desired Feather file path

csv_to_feather(csv_file_path, feather_file_path)
