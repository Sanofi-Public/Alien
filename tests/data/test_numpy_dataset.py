"""Test NumpyDataset class."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pytest

from alien.data import Dataset, NumpyDataset, TeachableDataset
from alien.utils import concatenate, join
from tests.conftest import N_FEATURES, N_SAMPLES


class TestNumpyDataset:
    """Test NumpyDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.data = np.vstack([self.X.T, self.y]).T

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db, NumpyDataset), "Dataset should be cast to NumpyDataset given numpy input."

    def test_extend(self):
        """Test shape and extend method."""
        # Test shape and extend method
        db = TeachableDataset.from_data(self.data)
        db.extend(self.data)
        expected_shape = (2 * N_SAMPLES, N_FEATURES + 1)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing"""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db[0], type(self.data))
        assert isinstance(db[:1], type(db))

    def test_join(self):
        """Test join helper function."""
        X, y = self.X, self.y
        db = TeachableDataset.from_data(self.data)
        db_join = join(db, db)
        assert isinstance(db_join, type(db)), "Joined dataset should be of same type as its constituents."
        expected_shape = (N_SAMPLES, 2 * (N_FEATURES + 1))
        assert db_join.shape == expected_shape
        assert (
            db_join[:2, X.shape[1] - 5 : X.shape[1] + 5]
            == np.concatenate([X[:2, -5:], y[:2].reshape(2, 1), X[:2, :4]], axis=1)
        ).all

    def test_concatenate(self):
        """Test concatenate helper function."""
        X = self.X
        db = TeachableDataset.from_data(self.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * N_SAMPLES, N_FEATURES + 1)
        assert db_concat.shape == expected_shape
        assert (db_concat[len(X) - 5 : len(X) + 5, :2] == np.concatenate([X[-5:, :2], X[:5, :2]], axis=0)).all()

    def test_reshape(self):
        """Test reshape method."""
        assert N_SAMPLES == 64
        assert N_FEATURES == 16
        data = Dataset(self.X)  # pylint: disable=abstract-class-instantiated
        data_0 = data.reshape(8, 8, 4, 4)
        assert data_0.shape == (8, 8, 4, 4)
        with pytest.raises(ValueError):
            _ = data.reshape(4, 8, 8, 4)
