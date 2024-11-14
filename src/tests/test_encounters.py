import unittest
import pandas as pd
import numpy as np
from utils.encounters import pairs_in_radius

class TestPairsInRadius(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe with latitude and longitude
        self.df = pd.DataFrame({
            'lat': [37.7749, 34.0522, 40.7128, 51.5074, 48.8566],
            'lon': [-122.4194, -118.2437, -74.0060, -0.1276, 2.3522]
        })

    def test_pairs_in_radius(self):
        # Test with a diameter that should include some pairs
        pairs, distances = pairs_in_radius(self.df, diam=5000000)  # 5000 km
        self.assertTrue(len(pairs) > 0)
        self.assertTrue(len(distances) > 0)
        self.assertEqual(len(pairs), len(distances))

    def test_no_pairs_in_radius(self):
        # Test with a very small diameter that should include no pairs
        pairs, distances = pairs_in_radius(self.df, diam=1)  # 1 meter
        self.assertEqual(len(pairs), 0)
        self.assertEqual(len(distances), 0)

    def test_all_pairs_in_radius(self):
        # Test with a very large diameter that should include all pairs
        pairs, distances = pairs_in_radius(self.df, diam=20000000)  # 20000 km
        self.assertTrue(len(pairs) > 0)
        self.assertTrue(len(distances) > 0)
        self.assertEqual(len(pairs), len(distances))

if __name__ == '__main__':
    unittest.main()