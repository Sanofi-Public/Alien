"""Tests for DeepChemDataset instances."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from alien.data import DeepChemDataset, TeachableDataset, as_DCDataset

dc = importorskip("deepchem")


class TestDeepChemDataset:
    """Test DeepChemDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup pytest fixture."""
        self.data_path = "tests/sample_data/sample_deepchem_data.csv"

    def test_from_data(self):
        """Test from_data method."""
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
        """Test from_csv method."""
        db = DeepChemDataset.from_csv(self.data_path, X="SMILES", y="y_exp")
        assert db.shape == (
            5,
            4,
        ), f"Expected shape (5, 4) from csv file. Got {db.shape}"

    def test_from_df(self):
        """Test from_df method."""
        df = pd.read_csv(self.data_path)
        db = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")
        assert db.shape[:2] == (5, 4), f"Expected shape (5, 4) from csv file. Got {db.shape}"

        df = df.rename({"SMILES": "X"}, axis=1)
        db = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")
        assert db.shape[:2] == (
            5,
            4,
        ), f"Expected shape (5, 4) from csv file. Got {db.shape}"

        df = df.rename({"X": "bad_name"}, axis=1)
        with pytest.raises(ValueError):
            _ = DeepChemDataset.from_df(df, X="SMILES", y="y_exp")


def test_as_dcdataset(get_X):
    """Test casting via as_DCDataset."""
    db = as_DCDataset(get_X)
    assert isinstance(db, dc.data.datasets.NumpyDataset)
