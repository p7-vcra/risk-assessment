import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import src.encounters.helper as helper

class TestVesselPairsInRadius(unittest.TestCase):

    def setUp(self):
        # Sample vessel data (latitude, longitude)
        df = pd.DataFrame({
            'Latitude': [55.00000, 55.00000, 55.00000, 55.00000, 55.00000],  # Base, 1.03km away, 10km away, 100km away, 1000km away
            'Longitude': [13.91570, 13.93200, 14.05620, 15.46300, 30.00000]
        })
        
        self.vessel_data = gpd.GeoDataFrame(
            df, 
            crs=4326, # this is the crs for lat/lon
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])
        )

        # Convert latitude and longitude to radians
        self.vessel_data['Latitude'] = np.deg2rad(self.vessel_data['Latitude'])
        self.vessel_data['Longitude'] = np.deg2rad(self.vessel_data['Longitude'])


    def test_get_vessel_pairs_in_radius_within_threshold(self):
        distance_threshold_in_km = 10 # 20km

        points, distances = helper.get_vessels_in_radius(self.vessel_data, distance_threshold_in_km)

        # Expected output with modified shape
        expected_points = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

        expected_distances = np.array([[0.0, 1.039594270818829, 8.960918455466697],
                            [0.0, 1.039594270818829, 7.9213246481542114],
                            [0.0, 7.9213246481542114, 8.960918455466697]]) # Distances in km between points

        # Assert that the returned pairs of points and distances are as expected
        for point_array, expected_array in zip(points, expected_points):
            np.testing.assert_array_equal(point_array, expected_array)

        for distance_array, expected_array in zip(distances, expected_distances):
            np.testing.assert_array_almost_equal(distance_array, expected_array, decimal=2)

    def test_get_vessel_pairs_in_radius_no_pairs(self):
        distance_threshold_in_km = 1  # 1 km

        points, distances = helper.get_vessels_in_radius(self.vessel_data, distance_threshold_in_km)

        self.assertEqual(len(points), 0)
        self.assertEqual(len(distances), 0)

    def test_get_vessel_pairs_in_radius_within_threshold_test_two(self):
        distance_threshold_in_km = 0.5

        data = {
            'Timestamp': ['10/09/2024 00:00:00'] * 7,
            'Type of mobile': ['Class A'] * 7,
            'MMSI': [218162000, 219004179, 219000872, 219000962, 219020188, 219765000, 219017664],
            'course' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Latitude': [57.146170, 56.788082, 69.000000, 56.891743, 56.786052, 57.183707, 56.974967],
            'Longitude': [8.537957, 8.875383, 0.000000, 9.164775, 8.876370, 8.648452, 8.922537]
        }
        
        df = pd.DataFrame(data)
        
        # Create a GeoDataFrame
        self.vessel_data = gpd.GeoDataFrame(
            df,
            crs=4326,  # CRS for lat/lon
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])
        )

        gdf = helper.convert_gdf_from_deg_to_rad(self.vessel_data)

        current_pairs = helper.get_vessel_pairs_in_radius(gdf, distance_threshold_in_km)

        expectedPair1 = (1, 4, 0.23359357978974854)
        expectedPair2 = (4, 1, 0.23359357978974854)

        for pair in current_pairs:
            self.assertTrue(pair == expectedPair1 or pair == expectedPair2)







if __name__ == '__main__':
    unittest.main()
