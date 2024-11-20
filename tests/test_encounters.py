import pytest
import pandas as pd
from src.utils.encounters import pairs_in_radius

@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe with latitude and longitude
    return pd.DataFrame({
        'lat': [37.7749, 34.0522, 40.7128, 51.5074, 48.8566],
        'lon': [-122.4194, -118.2437, -74.0060, -0.1276, 2.3522]
    })

class TestPairsInRadius():
    def test_pairs_in_radius(self, sample_dataframe):
        # Test with a diameter that should include some pairs
        pairs, distances = pairs_in_radius(sample_dataframe, diam=5000000)  # 5000 km
        assert len(pairs) > 0
        assert len(distances) > 0
        assert len(pairs) == len(distances)

    def test_no_pairs_in_radius(self, sample_dataframe):
        # Test with a very small diameter that should include no pairs
        pairs, distances = pairs_in_radius(sample_dataframe, diam=1)  # 1 meter
        assert len(pairs) == 0
        assert len(distances) == 0

    def test_all_pairs_in_radius(self, sample_dataframe):
        # Test with a very large diameter that should include all pairs
        pairs, distances = pairs_in_radius(sample_dataframe, diam=20000000)  # 20000 km
        assert len(pairs) > 0
        assert len(distances) > 0
        assert len(pairs) == len(distances)