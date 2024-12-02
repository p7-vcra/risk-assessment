import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import webbrowser
from asyncio import sleep
from datetime import datetime
from geopy.distance import geodesic
from sklearn.neighbors import BallTree


def plot_vessels_on_map(gdf, distance, specificIndexesToPlot=None):

    # Create a base map centered around the mean latitude and longitude
    # m = folium.Map(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', attr='Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri' ,location=[gdf['Latitude'].mean(), gdf['Longitude'].mean()], zoom_start=6)
    m = folium.Map(
        location=[gdf["Latitude"].mean(), gdf["Longitude"].mean()], zoom_start=6
    )

    if specificIndexesToPlot is not None:
        gdf = gdf.iloc[specificIndexesToPlot[0]]
        drawLineWithDistanceBetweenPoints(gdf, m)
        drawSafeRadiusForVessel(gdf, m, distance)

    # Plot the paths of the vessels
    for name, group in gdf.groupby("MMSI"):
        # Add a line for the path
        locations = group[["Latitude", "Longitude"]].values.tolist()
        folium.PolyLine(locations=locations, color="blue").add_to(m)

    # Plot the latest location of each vessel
    latest_locations = gdf.loc[gdf.groupby("MMSI")["# Timestamp"].idxmax()]
    for index, row in latest_locations.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"MMSI: {row['MMSI']}",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    # Save the map to an HTML file
    m.save("vessel_map.html")
    # open the map in the browser


def drawLineWithDistanceBetweenPoints(gdf, map_object):
    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            point1 = gdf.iloc[i]
            point2 = gdf.iloc[j]

            # Extract latitude and longitude from the points
            lat1, lon1 = point1["Latitude"], point1["Longitude"]
            lat2, lon2 = point2["Latitude"], point2["Longitude"]

            # Calculate the distance between the two points
            distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers

            # Add a line between the two points on the map
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)], color="green"
            ).add_to(map_object)

            # Add a marker at the midpoint with the distance information
            mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
            folium.Marker(
                location=[mid_lat, mid_lon],
                popup=f"Distance: {distance:.2f} km",
                icon=folium.Icon(color="green"),
            ).add_to(map_object)


def drawSafeRadiusForVessel(gdf, map_object, distance):
    for i in range(len(gdf)):
        point = gdf.iloc[i]

        # Extract latitude and longitude from the point
        lat, lon = point["Latitude"], point["Longitude"]

        # Add a circle around the point with a radius of 1 km
        folium.Circle(
            location=(lat, lon), radius=(distance * 1000), color="red", fill=True
        ).add_to(map_object)
