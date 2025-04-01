"""Tests for DictDataset class."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pandas as pd
import pytest

from alien.data import Dataset, DictDataset, NumpyDataset, TeachableDataset
from alien.utils import concatenate, join
from tests.conftest import N_FEATURES, N_SAMPLES


class TestDictDataset:
    """Test DictDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.data = {"X": self.X, "y": self.y}

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db, DictDataset), "Dataset should be cast to DictDataset given dictionary input."
        assert isinstance(db.X, NumpyDataset), "type of X attribute should be NumpyDataset since X is a numpy array"

    def test_from_pandas(self):
        """Test that a pandas dataframe is cast to DictDataset"""
        rows = [{"X": 1, "y": 1}, {"X": 2, "y": 2}]
        data = pd.DataFrame(rows)
        db = TeachableDataset.from_data(data)
        assert isinstance(db, DictDataset)

    def test_extend(self):
        """Test extend method."""
        db = TeachableDataset.from_data(self.data)
        assert db.X.shape == self.X.shape
        db.extend(self.data)
        expected_shape = (2 * N_SAMPLES, 2, N_FEATURES)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing"""
        db = TeachableDataset.from_data(self.data)
        # assert isinstance(db[0], type(TestDictDataset.data)), "Type of single index should be Dict"
        assert isinstance(db[:2], DictDataset), "Type of slice should be DictDataset"
        assert (db[:2, "X", :3] == self.X[:2, :3]).all()

    def test_set_attr(self):
        """Test setting a new attribute"""
        db = TeachableDataset.from_data(self.data)
        db.t = self.y + 1
        assert db[10, "t"] == (self.y + 1)[10]

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(self.data)
        _ = join(db, db)

    def test_concatenate(self):
        """Test concatenate helper function."""
        X = self.X
        db = TeachableDataset.from_data(self.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * N_SAMPLES, 2, N_FEATURES)
        assert db_concat.shape == expected_shape
        assert (db_concat[len(X) - 5 : len(X) + 5, "X", :2] == np.concatenate([X[-5:, :2], X[:5, :2]], axis=0)).all()

    def test_reshape(self):
        """Test reshape method."""
        X = self.X
        data = Dataset({"X1": X, "X2": X})  # pylint: disable=abstract-class-instantiated
        data_0 = data.reshape(8, 8, 2, -1)
        assert data_0.shape == (8, 8, 2, 16)
        with pytest.raises(ValueError):
            _ = data.reshape(4, 8, 2, 8, 4)
        with pytest.raises(ValueError):
            _ = data.reshape(8, 8, 4, 4, 2)
