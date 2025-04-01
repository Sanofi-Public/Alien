"""Tests for ObjectDataset instances."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pytest

from alien.data import NumpyDataset, TeachableDataset
from alien.utils import concatenate, join


class TestObjectDataset:
    """Test ObjectDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup pytest fixture."""
        self.data = np.array([np.array([1, 2, 3, 4]), np.array([5, 6]), np.array([7, 8, 9])], dtype=object)
        self.max_len = max([s.size for s in self.data])

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db, NumpyDataset), "Dataset should be cast to NumpyDataset given numpy input."

    def test_extend(self):
        """Test shape and extend method."""
        # Test shape and extend method
        db = TeachableDataset.from_data(self.data)
        db.extend(self.data)
        expected_shape = (2 * len(self.data), None)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing"""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db[0], self.data[0].__class__), "Indexing should return same type as input data."
        assert isinstance(db[:1], type(db))

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(self.data)
        db_join = join(db, db)
        assert isinstance(db_join, type(db)), "Joined dataset should be of same type as its constituents."
        assert db_join[0].shape == (2 * db[0].shape[0],)

    def test_concatenate(self):
        """Test concatenate helper function."""
        db = TeachableDataset.from_data(self.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * len(db.data), None)
        assert db_concat.shape == expected_shape
