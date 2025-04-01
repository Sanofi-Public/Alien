"""Test ShuffleDataset instances."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pytest

from alien.data import TeachableDataset
from alien.data.dataset import ShuffledDataset
from tests.conftest import N_FEATURES, N_SAMPLES


class TestShuffleDataset:
    """Test ShuffleDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.data = np.vstack([self.X.T, self.y]).T

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(self.data, shuffle=True)
        assert isinstance(db, ShuffledDataset), "db should be cast to ShuffledDataset if shuffle=True."

    def test_shuffle(self):
        """Test that db is shuffled."""
        data = self.data
        db = TeachableDataset.from_data(data, shuffle=True)
        assert db.shape == (N_SAMPLES, N_FEATURES + 1)
        assert (db.X != data).any(), "db should be shuffled."
        assert all(row.tolist() in data.tolist() for row in db.X), "Shuffle should contain the same rows."
