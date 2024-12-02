import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import itertools
from sklearn.neighbors import BallTree
import src.encounters.helper as helper


class TestHelperFunctions(unittest.TestCase):

    def setUp(self):
        # Sample vessel data (latitude, longitude)
        df = pd.DataFrame(
            {
                "Latitude": [
                    55.00000,
                    55.00000,
                    55.00000,
                    55.00000,
                    55.00000,
                ],  # Base, 1.03km away, 10km away, 100km away, 1000km away
                "Longitude": [13.91570, 13.93200, 14.05620, 15.46300, 30.00000],
            }
        )

        self.vessel_data = gpd.GeoDataFrame(
            df,
            crs=4326,  # this is the crs for lat/lon
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        )

        # Convert latitude and longitude to radians
        self.vessel_data["Latitude"] = np.deg2rad(self.vessel_data["Latitude"])
        self.vessel_data["Longitude"] = np.deg2rad(self.vessel_data["Longitude"])

        self.MMSI_data = pd.DataFrame(
            {
                "MMSI": ["123456789", "987654321", "112233445", "556677889"],
                "Latitude": [55.00000, 55.00000, 55.00000, 55.00000],
                "Longitude": [13.91570, 13.93200, 14.05620, 15.46300],
                "course": [0, 90, 180, 270],  # Placeholder courses
                "speed": [10, 20, 30, 40],
                "Length": [100, 200, 300, 400],
            }
        )

    def test_get_pairs(self):
        pairs = [(0, 1, 1.0), (0, 2, 8.0), (1, 3, 2.0)]
        timestamp = "2021-01-01 10:20:30"

        expected_output = pd.DataFrame(
            {
                "vessel_1": [
                    self.MMSI_data.iloc[0]["MMSI"],
                    self.MMSI_data.iloc[0]["MMSI"],
                    self.MMSI_data.iloc[1]["MMSI"],
                ],
                "vessel_2": [
                    self.MMSI_data.iloc[1]["MMSI"],
                    self.MMSI_data.iloc[2]["MMSI"],
                    self.MMSI_data.iloc[3]["MMSI"],
                ],
                "distance": [1.0, 8.0, 2.0],
                "start_time": [timestamp] * 3,
                "end_time": [timestamp] * 3,
                "vessel_1_longitude": [
                    self.MMSI_data.iloc[0]["Longitude"],
                    self.MMSI_data.iloc[0]["Longitude"],
                    self.MMSI_data.iloc[1]["Longitude"],
                ],
                "vessel_2_longitude": [
                    self.MMSI_data.iloc[1]["Longitude"],
                    self.MMSI_data.iloc[2]["Longitude"],
                    self.MMSI_data.iloc[3]["Longitude"],
                ],
                "vessel_1_latitude": [
                    self.MMSI_data.iloc[0]["Latitude"],
                    self.MMSI_data.iloc[0]["Latitude"],
                    self.MMSI_data.iloc[1]["Latitude"],
                ],
                "vessel_2_latitude": [
                    self.MMSI_data.iloc[1]["Latitude"],
                    self.MMSI_data.iloc[2]["Latitude"],
                    self.MMSI_data.iloc[3]["Latitude"],
                ],
                "vessel_1_speed": [
                    self.MMSI_data.iloc[0]["speed"],
                    self.MMSI_data.iloc[0]["speed"],
                    self.MMSI_data.iloc[1]["speed"],
                ],
                "vessel_2_speed": [
                    self.MMSI_data.iloc[1]["speed"],
                    self.MMSI_data.iloc[2]["speed"],
                    self.MMSI_data.iloc[3]["speed"],
                ],
                "vessel_1_course": [
                    self.MMSI_data.iloc[0]["course"],
                    self.MMSI_data.iloc[0]["course"],
                    self.MMSI_data.iloc[1]["course"],
                ],
                "vessel_2_course": [
                    self.MMSI_data.iloc[1]["course"],
                    self.MMSI_data.iloc[2]["course"],
                    self.MMSI_data.iloc[3]["course"],
                ],
                "vessel_1_length": [
                    self.MMSI_data.iloc[0]["Length"],
                    self.MMSI_data.iloc[0]["Length"],
                    self.MMSI_data.iloc[1]["Length"],
                ],
                "vessel_2_length": [
                    self.MMSI_data.iloc[1]["Length"],
                    self.MMSI_data.iloc[2]["Length"],
                    self.MMSI_data.iloc[3]["Length"],
                ],
            }
        ).set_index(["vessel_1", "vessel_2"])

        pairs_with_MMSI_info = helper.get_MMSI_info_for_current_pairs(
            timestamp, pairs, self.MMSI_data
        )

        print(expected_output)
        print(pairs_with_MMSI_info)

        for col in expected_output.columns:
            np.testing.assert_array_equal(
                expected_output[col], pairs_with_MMSI_info[col]
            )


if __name__ == "__main__":
    unittest.main()
