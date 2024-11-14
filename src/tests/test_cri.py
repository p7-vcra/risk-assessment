import unittest

import numpy as np
import pandas as pd
import geopandas as gpd
from utils.cri import EPS, calc_collision_eta, calc_safety_domain, cpa_membership, rel_bearing_membership, speed_ratio_membership
from utils.cri import calc_crit_dist

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

class TestCPAMembership(unittest.TestCase):
    def test_cpa_membership_below_min(self):
        self.assertEqual(cpa_membership(0.5, 1, 2), 1)

    def test_cpa_membership_at_min(self):
        self.assertEqual(cpa_membership(1, 1, 2), 1)

    def test_cpa_membership_between_min_and_max(self):
        self.assertAlmostEqual(cpa_membership(1.5, 1, 2), 0.25)

    def test_cpa_membership_at_max(self):
        self.assertEqual(cpa_membership(2, 1, 2), 0)

    def test_cpa_membership_above_max(self):
        self.assertEqual(cpa_membership(2.5, 1, 2), 0)


class TestCalcCollisionEta(unittest.TestCase):
    def test_calc_collision_eta_dcpa_below_d1(self):
        t1, t2 = calc_collision_eta(0.5, 10, 1, 2)
        self.assertAlmostEqual(t1, np.sqrt(1 ** 2 - 0.5 ** 2) / 10)
        self.assertAlmostEqual(t2, np.sqrt(2 ** 2 - 0.5 ** 2) / 10)

    def test_calc_collision_eta_dcpa_equal_d1(self):
        t1, t2 = calc_collision_eta(1, 10, 1, 2)
        self.assertAlmostEqual(t1, np.sqrt(1 ** 2 - 1 ** 2) / 10)
        self.assertAlmostEqual(t2, np.sqrt(2 ** 2 - 1 ** 2) / 10)

    def test_calc_collision_eta_dcpa_above_d1(self):
        t1, t2 = calc_collision_eta(1.5, 10, 1, 2)
        self.assertAlmostEqual(t1, (1 - 1.5) / 10)
        self.assertAlmostEqual(t2, np.sqrt(2 ** 2 - 1.5 ** 2) / 10)

    def test_calc_collision_eta_dcpa_equal_d2(self):
        t1, t2 = calc_collision_eta(2, 10, 1, 2)
        self.assertAlmostEqual(t1, (1 - 2) / 10)
        self.assertAlmostEqual(t2, np.sqrt(2 ** 2 - 2 ** 2) / 10)

    def test_calc_collision_eta_dcpa_above_d2(self):
        t1, t2 = calc_collision_eta(2.5, 10, 1, 2)
        self.assertAlmostEqual(t1, (1 - 2.5) / 10)
        self.assertTrue(t2, np.sqrt(2 ** 2 - 2.5 ** 2) / 10)


class TestCalcCritDist(unittest.TestCase):
    def test_calc_crit_dist_zero_length(self):
        crit_safe_dist, avoidance_measure_dist = calc_crit_dist(0, np.pi/4)
        self.assertEqual(crit_safe_dist, 0)
        self.assertAlmostEqual(avoidance_measure_dist, 1.7 * np.cos(np.pi/4 - np.deg2rad(19)) + np.sqrt(4.4 + 2.89 * np.cos(np.pi/4 - np.deg2rad(19)) ** 2))

    def test_calc_crit_dist_non_zero_length(self):
        crit_safe_dist, avoidance_measure_dist = calc_crit_dist(100, np.pi/4)
        self.assertEqual(crit_safe_dist, 100 * 12)
        self.assertAlmostEqual(avoidance_measure_dist, 1.7 * np.cos(np.pi/4 - np.deg2rad(19)) + np.sqrt(4.4 + 2.89 * np.cos(np.pi/4 - np.deg2rad(19)) ** 2))

    def test_calc_crit_dist_zero_bearing(self):
        crit_safe_dist, avoidance_measure_dist = calc_crit_dist(100, 0)
        self.assertEqual(crit_safe_dist, 100 * 12)
        self.assertAlmostEqual(avoidance_measure_dist, 1.7 * np.cos(0 - np.deg2rad(19)) + np.sqrt(4.4 + 2.89 * np.cos(0 - np.deg2rad(19)) ** 2))

    def test_calc_crit_dist_high_bearing(self):
        crit_safe_dist, avoidance_measure_dist = calc_crit_dist(100, np.pi)
        self.assertEqual(crit_safe_dist, 100 * 12)
        self.assertAlmostEqual(avoidance_measure_dist, 1.7 * np.cos(np.pi - np.deg2rad(19)) + np.sqrt(4.4 + 2.89 * np.cos(np.pi - np.deg2rad(19)) ** 2))

    def test_calc_crit_dist_negative_bearing(self):
        crit_safe_dist, avoidance_measure_dist = calc_crit_dist(100, -np.pi/4)
        self.assertEqual(crit_safe_dist, 100 * 12)
        self.assertAlmostEqual(avoidance_measure_dist, 1.7 * np.cos(-np.pi/4 - np.deg2rad(19)) + np.sqrt(4.4 + 2.89 * np.cos(-np.pi/4 - np.deg2rad(19)) ** 2))


class TestRelBearingMembership(unittest.TestCase):
    def test_rel_bearing_membership_zero_bearing(self):
        self.assertAlmostEqual(rel_bearing_membership(0), 1/2 * (np.cos(-np.deg2rad(19)) + np.sqrt(440/289 + np.cos(-np.deg2rad(19)) ** 2)) - 5/17)

    def test_rel_bearing_membership_positive_bearing(self):
        self.assertAlmostEqual(rel_bearing_membership(np.pi/4), 1/2 * (np.cos(np.pi/4 - np.deg2rad(19)) + np.sqrt(440/289 + np.cos(np.pi/4 - np.deg2rad(19)) ** 2)) - 5/17)

    def test_rel_bearing_membership_negative_bearing(self):
        self.assertAlmostEqual(rel_bearing_membership(-np.pi/4), 1/2 * (np.cos(-np.pi/4 - np.deg2rad(19)) + np.sqrt(440/289 + np.cos(-np.pi/4 - np.deg2rad(19)) ** 2)) - 5/17)

    def test_rel_bearing_membership_high_bearing(self):
        self.assertAlmostEqual(rel_bearing_membership(np.pi), 1/2 * (np.cos(np.pi - np.deg2rad(19)) + np.sqrt(440/289 + np.cos(np.pi - np.deg2rad(19)) ** 2)) - 5/17)

    def test_rel_bearing_membership_low_bearing(self):
        self.assertAlmostEqual(rel_bearing_membership(-np.pi), 1/2 * (np.cos(-np.pi - np.deg2rad(19)) + np.sqrt(440/289 + np.cos(-np.pi - np.deg2rad(19)) ** 2)) - 5/17)


class TestSpeedRatioMembership(unittest.TestCase):
    def test_speed_ratio_membership_equal_speeds(self):
        self.assertAlmostEqual(speed_ratio_membership(10, 10, np.pi/4), 1/(1 + 2/(1 * np.sqrt(1 ** 2 + 1 + 2 * 1 * np.sin(np.pi/4)) + EPS)))

    def test_speed_ratio_membership_target_faster(self):
        self.assertAlmostEqual(speed_ratio_membership(10, 20, np.pi/4), 1/(1 + 2/(2 * np.sqrt(2 ** 2 + 1 + 2 * 2 * np.sin(np.pi/4)) + EPS)))

    def test_speed_ratio_membership_own_faster(self):
        self.assertAlmostEqual(speed_ratio_membership(20, 10, np.pi/4), 1/(1 + 2/(0.5 * np.sqrt(0.5 ** 2 + 1 + 2 * 0.5 * np.sin(np.pi/4)) + EPS)))

    def test_speed_ratio_membership_zero_own_speed(self):
        with self.assertRaises(ZeroDivisionError):
            speed_ratio_membership(0, 10, np.pi/4)

    def test_speed_ratio_membership_zero_target_speed(self):
        self.assertAlmostEqual(speed_ratio_membership(10, 0, np.pi/4), 1/(1 + 2/(0 * np.sqrt(0 ** 2 + 1 + 2 * 0 * np.sin(np.pi/4)) + EPS)))

    def test_speed_ratio_membership_zero_relative_course(self):
        self.assertAlmostEqual(speed_ratio_membership(10, 10, 0), 1/(1 + 2/(1 * np.sqrt(1 ** 2 + 1 + 2 * 1 * np.sin(0)) + EPS)))

    def test_speed_ratio_membership_high_relative_course(self):
        self.assertAlmostEqual(speed_ratio_membership(10, 10, np.pi), 1/(1 + 2/(1 * np.sqrt(1 ** 2 + 1 + 2 * 1 * np.sin(np.pi)) + EPS)))


class TestCalcSafetyDomain(unittest.TestCase):
    def test_calc_safety_domain_first_interval(self):
        d1, d2 = calc_safety_domain(np.pi / 4)
        self.assertAlmostEqual(d1, 1.1 - 0.2 * (np.pi / 4) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.1 - 0.2 * (np.pi / 4) / np.pi))

    def test_calc_safety_domain_second_interval(self):
        d1, d2 = calc_safety_domain(3 * np.pi / 4)
        self.assertAlmostEqual(d1, 1.0 - 0.4 * (3 * np.pi / 4) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.0 - 0.4 * (3 * np.pi / 4) / np.pi))

    def test_calc_safety_domain_third_interval(self):
        d1, d2 = calc_safety_domain(5 * np.pi / 4)
        self.assertAlmostEqual(d1, 1.0 - 0.4 * (2 * np.pi - 5 * np.pi / 4) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.0 - 0.4 * (2 * np.pi - 5 * np.pi / 4) / np.pi))

    def test_calc_safety_domain_fourth_interval(self):
        d1, d2 = calc_safety_domain(7 * np.pi / 4)
        self.assertAlmostEqual(d1, 1.1 - 0.4 * (2 * np.pi - 7 * np.pi / 4) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.1 - 0.4 * (2 * np.pi - 7 * np.pi / 4) / np.pi))

    def test_calc_safety_domain_boundary_values(self):
        d1, d2 = calc_safety_domain(0)
        self.assertAlmostEqual(d1, 1.1)
        self.assertAlmostEqual(d2, 2.2)

        d1, d2 = calc_safety_domain(5 * np.pi / 8)
        self.assertAlmostEqual(d1, 1.0 - 0.4 * (5 * np.pi / 8) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.0 - 0.4 * (5 * np.pi / 8) / np.pi))

        d1, d2 = calc_safety_domain(np.pi)
        self.assertAlmostEqual(d1, 1.0 - 0.4 * (2 * np.pi - np.pi) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.0 - 0.4 * (2 * np.pi - np.pi) / np.pi))

        d1, d2 = calc_safety_domain(11 * np.pi / 8)
        self.assertAlmostEqual(d1, 1.1 - 0.4 * (2 * np.pi - (11 * np.pi / 8)) / np.pi)
        self.assertAlmostEqual(d2, 2 * (1.1 - 0.4 * (2 * np.pi - (11 * np.pi / 8)) / np.pi))

        d1, d2 = calc_safety_domain(2 * np.pi)
        self.assertTrue(np.isnan(d1))
        self.assertTrue(np.isnan(d2))


if __name__ == '__main__':
    unittest.main()
