"""Tests for TupleDataset instances."""

# pylint: disable=attribute-defined-outside-init
import pytest

from alien.data import Dataset, NumpyDataset, TeachableDataset, TupleDataset
from alien.utils import concatenate, join
from tests.conftest import N_FEATURES, N_SAMPLES


class TestTupleDataset:
    """Test TupleDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.data = (self.X, self.y)

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(self.data)

        assert isinstance(db, TupleDataset), "Dataset should be cast to TupleDataset given tuple input"
        assert isinstance(db[:,0], NumpyDataset), "sub-dataset should be NumpyDataset"

    def test_shape_extend(self):
        """Test shape and extend method."""
        db = TeachableDataset.from_data(self.data)
        db.extend(self.data)
        expected_shape = (2 * N_SAMPLES, 2, N_FEATURES)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing."""
        db = TeachableDataset.from_data(self.data)
        assert isinstance(db[0], type(self.data))
        assert db[0, 1] == self.y[0]
        assert (db[:2, 0, :2] == self.X[:2, :2]).all()

    def test_properties(self):
        """Test properties."""
        db = TeachableDataset.from_data(self.data)
        assert db.tuple is db.data

    def test_find(self):
        """Test find method."""
        db = TeachableDataset.from_data(self.data)
        target_ix = 37
        curr_ix = db.find((self.X[target_ix], self.y[target_ix]))
        assert curr_ix == target_ix

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(self.data)
        db_join = join(db, db)
        assert isinstance(db_join, type(db)), "Joined dataset should be of same type as its constituents."
        expected_shape = (N_SAMPLES, 4, N_FEATURES)
        assert db_join.shape == expected_shape
        # Here the join happens along the second dimension (hence the 4 in expected shape above)
        # so don't need to test indexing along an axis.

    def test_concatenate(self):
        """Test concatenate helper function."""
        db = TeachableDataset.from_data(self.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * db.shape[0], *db.shape[1:])
        assert db_concat.shape == expected_shape

    def test_reshape(self):
        """Test reshape method."""
        assert N_SAMPLES == 64
        assert N_FEATURES == 16
        data = Dataset((self.X, self.X))  # pylint: disable=abstract-class-instantiated
        data_0 = data.reshape(8, 8, 2, -1)
        assert data_0.shape == (8, 8, 2, 16)
        with pytest.raises(ValueError):
            _ = data.reshape(4, 8, 2, 8, 4)
        with pytest.raises(ValueError):
            _ = data.reshape(8, 8, 4, 4, 2)
