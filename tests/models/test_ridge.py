"""Tests for the BayesianRidgeRegressor class."""

# pylint: disable=attribute-defined-outside-init
import numpy as np
import pytest

from alien.models import BayesianRidgeRegressor
from tests.conftest import N_FEATURES, SEED


class TestBayesianRidgeRegressor:
    """Test BayesianRidgeRegressor class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = BayesianRidgeRegressor(X=self.X, y=self.y, random_seed=SEED)
        self.regressor.fit()

    def test_covariance(self):
        """Test the model's covariance method."""
        regressor = self.regressor
        cov = regressor.covariance(self.X)
        expected_shape = (self.X.shape[0], self.X.shape[0])
        assert cov.shape == expected_shape, f"Covariance should be a square matrix of of shape {expected_shape}."

        zero_matrix = np.zeros(self.X.shape)
        cov_0 = regressor.covariance(zero_matrix)
        assert (cov_0 == 0).all(), "Covariance of zero matrix should be zero."

    def test_predict(self):
        """Test the model's prediction."""
        regressor = self.regressor
        preds = regressor.predict(self.X)
        assert preds.shape == self.y.shape, f"Expected predictions of shape {self.y.shape}"
        preds = regressor.predict(self.X)
        assert preds.shape == self.y.shape, f"Expected predictions of shape {self.y.shape}"

        preds_0 = regressor.predict(np.zeros(self.X.shape))
        b = regressor.model.intercept_
        assert (preds_0 == b).all(), "Prediction of zero should equal the bias term."

    def test_embedding(self):
        """
        Test embedding method.
        In this instance it should be the identity since
        the "last layer" is the model itself."""
        regressor = self.regressor
        embds = regressor.embedding(self.X)
        assert (self.X == embds).all(), "Embeddings for a linear model should be equal to the input."

    def test_properties(self):
        """Test properties of the model."""
        regressor = self.regressor
        weights = regressor.weights
        assert weights.shape == (N_FEATURES,)
        bias = regressor.bias
        assert bias.shape == ()
        _, _ = regressor.linearization()
