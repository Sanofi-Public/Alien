"""Test Dataset class."""

import pytest

from alien.data import NumpyDataset, TeachableDataset


class TestDataset:
    """Test base Dataset class"""

    def test_from_data(self):
        """Test from_data method."""
        data = [1, 2]
        db = TeachableDataset.from_data(data, convert_sequences=True)
        assert isinstance(db, NumpyDataset)
        db = TeachableDataset.from_data(data, convert_sequences=False)
        assert isinstance(db, TeachableDataset)

        with pytest.warns(UserWarning):
            db = TeachableDataset.from_data("x=[1,2]")
            assert isinstance(db, TeachableDataset)
