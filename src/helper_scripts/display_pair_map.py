import pandas as pd
import folium
import numpy as np
import find_ship_trace
import matplotlib.pyplot as plt

# Load the CSV file
CLUSTER_PATH = './output/output_20241125-12:38:30_cutdown.csv'
DATA_PATH = './data/RealData.csv'

cluster_df = pd.read_csv(CLUSTER_PATH)

# Get unique MMSIs
mmsis = pd.concat([cluster_df['vessel_1'], cluster_df['vessel_2']]).unique()

# Filter the data (custom filtering function)
df = find_ship_trace.filter_mmsi_data(DATA_PATH, mmsis, cluster_df)

# Create a map centered at the mean latitude and longitude
mean_lat = df['Latitude'].mean()
mean_lon = df['Longitude'].mean()
mymap = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# Define a function to generate distinct colors
def generate_colors(n):
    # Use a colormap to generate unique colors
    cmap = plt.get_cmap('tab20')
    return [cmap(i % 20)[:3] for i in range(n)]  # Repeat colors if > 20

# Normalize RGB values to hex format for Folium
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

# Generate colors for the ships
colors = [rgb_to_hex(color) for color in generate_colors(len(mmsis))]

# Function to plot a ship's trajectory
def plot_trajectory(mmsi, color, index, total):
    print(f"Processing MMSI {index + 1} out of {total}: {mmsi}")
    # Filter the data for the ship's MMSI
    ship_data = df[df['MMSI'] == mmsi]
    
    # Get the coordinates of the ship's trajectory
    coordinates = list(zip(ship_data['Latitude'], ship_data['Longitude']))
    
    # Plot the trajectory using a PolyLine
    folium.PolyLine(coordinates, color=color, weight=2.5, opacity=1).add_to(mymap)

# Plot the trajectories for all ships
if len(mmsis) > 0:
    for i, mmsi in enumerate(mmsis):
        plot_trajectory(mmsi, colors[i], i, len(mmsis))
else:
    print("Error: No ships found.")

# Save the map to an HTML file
mymap.save("ship_trajectories.html")

# To display the map inline (if using Jupyter):
# mymap
