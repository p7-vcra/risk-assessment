import os
import sys
import unittest
import pandas as pd
import src.helper as helper
import src.data_util as data_utils


class TestUpdatePairs(unittest.TestCase):

    def test_update_pairs_returns_correct_pairs(self):
        # Active pairs DataFrame
        active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        ).astype(data_utils.pairType)

        # Current pairs DataFrame
        current_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [0.5],
                "start_time": [pd.Timestamp("2023-01-01 00:02:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:03:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        ).astype(data_utils.pairType)

        expected_active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [0.5],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:03:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        ).astype(data_utils.pairType)

        timestamp = pd.Timestamp("2023-01-01 00:03:00")

        updated_active_pairs, updated_inactive_pairs = helper.update_pairs(
            timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds=60
        )

        pd.testing.assert_frame_equal(updated_active_pairs, expected_active_pairs)
        self.assertEqual(len(updated_inactive_pairs), 0)  # No pairs should be inactive

    def test_update_pairs_no_current_pairs(self):
        # Active pairs DataFrame with one pair
        active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180.0],
                "vessel_2_course": [200.0],
                "vessel_1_length": [50.0],
                "vessel_2_length": [60.0],
            }
        ).set_index(["vessel_1", "vessel_2"])

        # No current pairs
        current_pairs = pd.DataFrame(
            columns=["vessel_1", "vessel_2", "distance", "start_time", "end_time", "vessel_1_longitude", 
                     "vessel_2_longitude", "vessel_1_latitude", "vessel_2_latitude", 
                     "vessel_1_speed", "vessel_2_speed", "vessel_1_course", "vessel_2_course", 
                     "vessel_1_length", "vessel_2_length"]
        ).set_index(["vessel_1", "vessel_2"])

        expected_active_pairs = pd.DataFrame(
            columns=["vessel_1", "vessel_2", "distance", "start_time", "end_time", "vessel_1_longitude", 
                     "vessel_2_longitude", "vessel_1_latitude", "vessel_2_latitude", 
                     "vessel_1_speed", "vessel_2_speed", "vessel_1_course", "vessel_2_course", 
                     "vessel_1_length", "vessel_2_length"]
        ).set_index(["vessel_1", "vessel_2"]).astype(data_utils.pairType)
        expected_inactive_pairs = active_pairs.copy()

        timestamp = pd.Timestamp("2023-01-01 00:02:00")

        updated_active_pairs, updated_inactive_pairs = helper.update_pairs(
            timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds=60
        )

        pd.testing.assert_frame_equal(updated_active_pairs, expected_active_pairs)
        pd.testing.assert_frame_equal(updated_inactive_pairs, expected_inactive_pairs)

    def test_update_pairs_with_no_distance_update(self):
        # Active pairs DataFrame
        active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        )

        # Current pairs with same distance
        current_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:02:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:03:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        )

        expected_active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:03:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        ).astype(data_utils.pairType)

        timestamp = pd.Timestamp("2023-01-01 00:03:00")

        updated_active_pairs, updated_inactive_pairs = helper.update_pairs(
            timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds=60
        )

        pd.testing.assert_frame_equal(updated_active_pairs, expected_active_pairs)
        self.assertEqual(len(updated_inactive_pairs), 0)  # No pairs should be inactive

    def test_update_pairs_with_temporal_threshold_exceeded(self):
        # Active pairs DataFrame
        active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        )

        # Current pairs with the same MMSI but with a larger distance
        current_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [5.0],
                "start_time": [pd.Timestamp("2023-01-01 00:02:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:02:30")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        )
        expected_inactive_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        ).astype(data_utils.pairType)

        timestamp = pd.Timestamp("2023-01-01 00:02:30")

        updated_active_pairs, updated_inactive_pairs = helper.update_pairs(
            timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds=15
        )

        #check that updated_Active_pairs is empty
        self.assertEqual(len(updated_active_pairs), 0)
        pd.testing.assert_frame_equal(updated_inactive_pairs, expected_inactive_pairs)


    def test_update_pairs_with_temporal_threshold_not_exceeded(self):
        # Active pairs DataFrame
        active_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [1.0],
                "start_time": [pd.Timestamp("2023-01-01 00:00:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:01:00")],
                "vessel_1_longitude": [10.0],
                "vessel_2_longitude": [20.0],
                "vessel_1_latitude": [30.0],
                "vessel_2_latitude": [40.0],
                "vessel_1_speed": [12.0],
                "vessel_2_speed": [14.0],
                "vessel_1_course": [180],
                "vessel_2_course": [200],
                "vessel_1_length": [50],
                "vessel_2_length": [60],
            }
        )

        # Current pairs with the same MMSI but with a larger distance
        current_pairs = pd.DataFrame(
            {
                "vessel_1": ["123456789"],
                "vessel_2": ["987654321"],
                "distance": [5.0],
                "start_time": [pd.Timestamp("2023-01-01 00:02:00")],
                "end_time": [pd.Timestamp("2023-01-01 00:02:30")],
                "vessel_1_longitude": [11.0],
                "vessel_2_longitude": [21.0],
                "vessel_1_latitude": [31.0],
                "vessel_2_latitude": [41.0],
                "vessel_1_speed": [13.0],
                "vessel_2_speed": [15.0],
                "vessel_1_course": [185],
                "vessel_2_course": [205],
                "vessel_1_length": [55],
                "vessel_2_length": [65],
            }
        )

        expected_active_pairs = pd.DataFrame(
            columns=["vessel_1", "vessel_2", "distance", "start_time", "end_time", "vessel_1_longitude", 
                     "vessel_2_longitude", "vessel_1_latitude", "vessel_2_latitude", 
                     "vessel_1_speed", "vessel_2_speed", "vessel_1_course", "vessel_2_course", 
                     "vessel_1_length", "vessel_2_length"]
        ).astype(data_utils.pairType)

        expected_inactive_pairs = pd.DataFrame(
            columns=["vessel_1", "vessel_2", "distance", "start_time", "end_time", "vessel_1_longitude", 
                     "vessel_2_longitude", "vessel_1_latitude", "vessel_2_latitude", 
                     "vessel_1_speed", "vessel_2_speed", "vessel_1_course", "vessel_2_course", 
                     "vessel_1_length", "vessel_2_length"]
        ).astype(data_utils.pairType)

        timestamp = pd.Timestamp("2023-01-01 00:02:30")
        
        updated_active_pairs, updated_inactive_pairs = helper.update_pairs(
            timestamp, active_pairs, current_pairs, temporal_threshold_in_seconds=200
        )

        updated_active_pairs = updated_active_pairs.reset_index(drop=True) # reset index due to weird mismatch between indexes
        updated_inactive_pairs = updated_inactive_pairs.reset_index(drop=True)

        pd.testing.assert_frame_equal(updated_active_pairs, expected_active_pairs)
        pd.testing.assert_frame_equal(updated_inactive_pairs, expected_inactive_pairs)
