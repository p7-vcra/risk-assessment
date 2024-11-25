import pandas as pd
import folium
import numpy as np

# Load the CSV file
data_path = './filtered_mmsi_data_no_new_line.csv'
df = pd.read_csv(data_path)

# Ensure proper data types
df['Timestamp'] = pd.to_datetime(df['# Timestamp'])
df['Latitude'] = pd.to_numeric(df['Latitude'])
df['Longitude'] = pd.to_numeric(df['Longitude'])

# Get unique MMSIs (for two ships)
mmsis = df['MMSI'].unique()

# Create a map centered at the mean latitude and longitude of both ships
mean_lat = df['Latitude'].mean()
mean_lon = df['Longitude'].mean()
mymap = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# Function to plot a ship's trajectory
def plot_trajectory(mmsi, color):
    # Filter the data for the ship's MMSI
    ship_data = df[df['MMSI'] == mmsi]
    
    # Get the coordinates of the ship's trajectory
    coordinates = list(zip(ship_data['Latitude'], ship_data['Longitude']))
    
    # Plot the trajectory using a PolyLine
    folium.PolyLine(coordinates, color=color, weight=2.5, opacity=1).add_to(mymap)

# Plot the trajectories of the two ships
if len(mmsis) > 1:
    plot_trajectory(mmsis[0], 'blue')  # First ship in blue
    plot_trajectory(mmsis[1], 'red')   # Second ship in red
else:
    print("Error: Less than two ships found.")

# Save the map to an HTML file
mymap.save("ship_trajectories.html")

# To display the map inline (if using Jupyter):
# mymap
