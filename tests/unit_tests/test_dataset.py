"""
Test casting with TeachableDataset.from_data(*), shapes, extend/append,
and indexing of different dataset types.
Fix n_samples, n_features, X and y for all tests.
"""
import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from alien.data import (
    Dataset,
    DeepChemDataset,
    DictDataset,
    NumpyDataset,
    TeachableDataset,
    TupleDataset,
    as_DCDataset,
)
from alien.data.dataset import ShuffledDataset
from alien.utils import concatenate, join

np.random.seed(0)
n_samples = 64
n_features = 16
X = np.random.normal(size=(n_samples, n_features))
y = np.random.normal(size=n_samples)


class TestDictDataset:
    data = {"X": X, "y": y}

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(TestDictDataset.data)
        assert isinstance(
            db, DictDataset
        ), "Dataset should be cast to DictDataset given dictionary input."
        assert isinstance(
            db.X, NumpyDataset
        ), "type of X attribute should be NumpyDataset since X is a numpy array"

    def test_from_pandas(self):
        """Test that a pandas dataframe is cast to DictDataset"""
        rows = [{"X": 1, "y": 1}, {"X": 2, "y": 2}]
        data = pd.DataFrame(rows)
        db = TeachableDataset.from_data(data)
        assert isinstance(db, DictDataset)

    def test_extend(self):
        """Test extend method."""
        db = TeachableDataset.from_data(TestDictDataset.data)
        assert db.X.shape == X.shape
        db.extend(TestDictDataset.data)
        expected_shape = (2 * n_samples, 2, n_features)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing"""
        db = TeachableDataset.from_data(TestDictDataset.data)
        assert isinstance(db[0], type(TestDictDataset.data)), "Type of single index should be Dict"
        assert isinstance(db[:1], DictDataset), "Type of slice should be DictDataset"
        assert (db[:2, "X", :3] == X[:2, :3]).all()

    def test_set_attr(self):
        """Test setting a new attribute"""
        db = TeachableDataset.from_data(TestDictDataset.data)
        db.t = y + 1
        assert db[10, "t"] == (y + 1)[10]

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(TestDictDataset.data)
        # TODO: join tests once implemented
        db_join = join(db, db)

    def test_concatenate(self):
        """Test concatenate helper function."""
        db = TeachableDataset.from_data(TestDictDataset.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * n_samples, 2, n_features)
        assert db_concat.shape == expected_shape
        assert (
            db_concat[len(X) - 5 : len(X) + 5, "X", :2]
            == np.concatenate([X[-5:, :2], X[:5, :2]], axis=0)
        ).all()

    def test_reshape(self):
        assert n_samples == 64
        assert n_features == 16
        data = Dataset({'X1':X, 'X2':X})
        data_0 = data.reshape(8, 8, 2, -1)
        assert data_0.shape == (8, 8, 2, 16)
        with pytest.raises(ValueError):
            data_1 = data.reshape(4, 8, 2, 8, 4)
        with pytest.raises(ValueError):
            data_2 = data.reshape(8, 8, 4, 4, 2)


class TestNumpyDataset:
    data = np.vstack([X.T, y]).T

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(TestNumpyDataset.data)
        assert isinstance(
            db, NumpyDataset
        ), "Dataset should be cast to NumpyDataset given numpy input."
        assert isinstance(
            db.X, NumpyDataset
        ), "type of X attribute should be NumpyDataset since X is a numpy array"

    def test_extend(self):
        """Test shape and extend method."""
        # Test shape and extend method
        db = TeachableDataset.from_data(TestNumpyDataset.data)
        db.extend(TestNumpyDataset.data)
        expected_shape = (2 * n_samples, n_features + 1)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing"""
        db = TeachableDataset.from_data(TestNumpyDataset.data)
        assert isinstance(db[0], type(TestNumpyDataset.data))
        assert (db.X[:2].data == db[:2, :-1].data).all()

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(TestNumpyDataset.data)
        db_join = join(db, db)
        assert isinstance(
            db_join, type(db)
        ), "Joined dataset should be of same type as its constituents."
        expected_shape = (n_samples, 2 * (n_features + 1))
        assert db_join.shape == expected_shape
        assert (
            db_join[:2, X.shape[1] - 5 : X.shape[1] + 5]
            == np.concatenate([X[:2, -5:], y[:2].reshape(2, 1), X[:2, :4]], axis=1)
        ).all

    def test_concatenate(self):
        """Test concatenate helper function."""
        db = TeachableDataset.from_data(TestNumpyDataset.data)
        db_concat = concatenate(db, db)
        expected_shape = (2 * n_samples, n_features + 1)
        assert db_concat.shape == expected_shape
        assert (
            db_concat[len(X) - 5 : len(X) + 5, :2]
            == np.concatenate([X[-5:, :2], X[:5, :2]], axis=0)
        ).all()

    def test_reshape(self):
        assert n_samples == 64
        assert n_features == 16
        data = Dataset(X)
        data_0 = data.reshape(8, 8, 4, 4)
        assert data_0.shape == (8, 8, 4, 4)
        with pytest.raises(ValueError):
            data_1 = data.reshape(4, 8, 8, 4)



class TestDataset:
    def test_from_data(self):
        data = [1, 2]
        db = TeachableDataset.from_data(data, convert_sequences=True)
        assert isinstance(db, NumpyDataset)
        db = TeachableDataset.from_data(data, convert_sequences=False)
        assert isinstance(db, TeachableDataset)

        with pytest.warns(UserWarning):
            db = TeachableDataset.from_data("x=[1,2]")
            assert isinstance(db, TeachableDataset)


class TestTupleDataset:
    data = (X, y)

    def test_casting(self):
        """Test that db is cast to the correct type given data."""
        db = TeachableDataset.from_data(TestTupleDataset.data)

        assert isinstance(
            db, TupleDataset
        ), "Dataset should be cast to TupleDataset given tuple input"
        assert isinstance(
            db.X, NumpyDataset
        ), "type of X attribute should be NumpyDataset since X is a numpy array"

    def test_shape_extend(self):
        """Test shape and extend method."""
        db = TeachableDataset.from_data(TestTupleDataset.data)
        db.extend(TestTupleDataset.data)
        expected_shape = (2 * n_samples, 2, n_features)
        assert db.shape == expected_shape, f"Expected shape {expected_shape}, got {db.shape}"

    def test_indexing(self):
        """Test indexing."""
        db = TeachableDataset.from_data(TestTupleDataset.data)
        assert isinstance(db[0], type(TestTupleDataset.data))
        assert db[0, 1] == y[0]
        assert (db[:2, 0, :2] == X[:2, :2]).all()

    def test_properties(self):
        """Test properties."""
        db = TeachableDataset.from_data(TestTupleDataset.data)
        assert db.tuple is db.data

    def test_find(self):
        db = TeachableDataset.from_data(TestTupleDataset.data)
        target_ix = 37
        ix = db.find((X[target_ix], y[target_ix]))
        assert ix == target_ix

    def test_join(self):
        """Test join helper function."""
        db = TeachableDataset.from_data(TestTupleDataset.data)
        db_join = join(db, db)
        assert isinstance(
            db_join, type(db)
        ), "Joined dataset should be of same type as its constituents."
        expected_shape = (n_samples, 4, n_features)
        assert db_join.shape == expected_shape
        # Here the join happens along the second dimension (hence the 4 in expected shape above)
        # so don't need to test indexing along an axis.

    def test_concatenate(self):
        """Test concatenate helper function."""
        db = TeachableDataset.from_data(TestTupleDataset.data)
        # TODO: concatenate doesn't work for tuples
        db_concat = concatenate(db, db)
        expected_shape = (2 * db.shape[0], *db.shape[1:])
        assert db_concat.shape == expected_shape

    def test_reshape(self):
        assert n_samples == 64
        assert n_features == 16
        data = Dataset((X,X))
        data_0 = data.reshape(8, 8, 2, -1)
        assert data_0.shape == (8, 8, 2, 16)
        with pytest.raises(ValueError):
            data_1 = data.reshape(4, 8, 2, 8, 4)
        with pytest.raises(ValueError):
            data_2 = data.reshape(8, 8, 4, 4, 2)


class TestShuffleDataset:
    data = np.vstack([X.T, y]).T

    def test_casting(self):
        db = TeachableDataset.from_data(TestShuffleDataset.data, shuffle=True)
        assert isinstance(
            db, ShuffledDataset
        ), "db should be cast to ShuffledDataset if shuffle=True."

    def test_shuffle(self):
        db = TeachableDataset.from_data(TestShuffleDataset.data, shuffle=True)
        assert db.shape == (n_samples, n_features + 1)
        assert (db.X != X).any(), "db should be shuffled."
        assert all(
            row.tolist() in X.tolist() for row in db.X
        ), "Shuffle should contain the same rows."

        old_ix = 0
        new_ix = np.where(np.all(db.X == X[old_ix], axis=1))[0][0]
        assert y[old_ix] == db.y[new_ix]


class TestDeepChemDataset:
    data_path = "tests/unit_tests/data/sample_deepchem_data.csv"

    def test_from_data(self):
        dc = importorskip("deepchem")
        dataset = dc.data.NumpyDataset(
            X=np.random.rand(5, 3),
            y=np.random.rand(
                5,
            ),
            ids=np.arange(5),
        )
        db = TeachableDataset.from_data(dataset)
        assert isinstance(db, DeepChemDataset)

    def test_from_csv(self):
        db = DeepChemDataset.from_csv(TestDeepChemDataset.data_path, X="SMILES", y="y_exp")
        assert db.shape == (
            5,
            4,
        ), f"Expected shape (5, 4) from csv file. Got {db.shape}"

    def test_from_df(self):
        df = pd.read_csv(TestDeepChemDataset.data_path)
        db = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")
        assert db.shape == (
            5,
            4,
        ), f"Expected shape (5, 4) from csv file. Got {db.shape}"

        df = df.rename({"SMILES": "X"}, axis=1)
        db = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")
        assert db.shape == (
            5,
            4,
        ), f"Expected shape (5, 4) from csv file. Got {db.shape}"

        df = df.rename({"X": "bad_name"}, axis=1)
        with pytest.raises(ValueError):
            _ = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")


def test_as_dcdataset():
    dc = importorskip("deepchem")
    db = as_DCDataset(X)
    assert isinstance(db, dc.data.datasets.NumpyDataset)
