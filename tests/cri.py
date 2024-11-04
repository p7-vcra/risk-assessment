import unittest

import pandas as pd
import geopandas as gpd

data = {
    "own_vessel_id": [1, 1, 1, 1, 1],
    "own_longitude": [37.45, 37.45, 37.45, 37.45, 37.45],
    "own_latitude": [2.06, 2.06, 2.06, 2.06, 2.06],
    "own_speed": [14.18, 14.18, 14.18, 14.18, 14.18],
    "own_heading": [218.72, 218.72, 218.72, 218.72, 218.72],
    "target_vessel_id": [2, 3, 4, 5, 6],
    "target_longitude": [95.07, 73.20, 59.87, 15.60, 15.60],
    "target_latitude": [96.99, 83.24, 21.23, 18.18, 18.34],
    "target_speed": [7.09, 9.38, 10.50, 11.84, 16.78],
    "target_heading": [61.39, 23.42, 341.60, 347.63, 291.02]
}

vessel_pair_data = pd.DataFrame(data)

features = ["vessel_id", "longitude", "latitude", "speed", "heading"]
own_features = ["own_vessel_id", "own_longitude", "own_latitude", "own_speed", "own_heading"]
target_features = ["target_vessel_id", "target_longitude", "target_latitude", "target_speed", "target_heading"]

own_vessels = vessel_pair_data[own_features]
target_vessels = vessel_pair_data[target_features]

own_vessels.columns = features
target_vessels.columns = features

own_vessels = gpd.GeoDataFrame(own_vessels, crs=4326, geometry=gpd.points_from_xy(own_vessels['longitude'], own_vessels['latitude']))
target_vessels = gpd.GeoDataFrame(target_vessels, crs=4326, geometry=gpd.points_from_xy(target_vessels['longitude'], target_vessels['latitude']))

class TestCri(unittest.TestCase):

    def test_calculate_cpa():
        pass

if __name__ == '__main__':
    print(own_vessels)
    unittest.main()
