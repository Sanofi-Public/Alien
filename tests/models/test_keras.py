"""Test KerasRegressor with dropout uncertainty."""

# pylint: disable=attribute-defined-outside-init
import pytest
from pytest import importorskip

from alien.models import KerasRegressor
from tests.conftest import ENSEMBLE_SIZE

keras = importorskip("keras")


class TestKerasMCDropoutRegressor:
    """Test KerasRegressor with dropout uncertainty."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_keras_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_keras_model
        self.regressor = KerasRegressor(
            model=self.model, X=self.X, y=self.y, ensemble_size=ENSEMBLE_SIZE, uncertainty="dropout"
        )

    def test_fit(self):
        """Test fit method"""
        self.regressor.fit(epochs=5)

    def test_ensemble(self):
        """Test ensemble prediction."""
        X = self.X
        ensemble_preds = self.regressor.predict_fixed_ensemble(X)
        expected_shape = (len(self.X), ENSEMBLE_SIZE, 1)
        assert ensemble_preds.shape == expected_shape

    def test_predict_value_error(self):
        """Test predict method raises ValueError."""
        X = self.X
        with pytest.raises(ValueError):
            self.regressor.predict_samples(X, n=ENSEMBLE_SIZE, dropout_seeds=[i for i in range(ENSEMBLE_SIZE - 1)])
    
    def test_entropy_stats(self):
        """Test entropy stats."""
        X = self.X
        mutual_info = self.regressor.mutual_info(X)
        assert mutual_info.shape == (len(X), len(X))
        batch_entropy = self.regressor.approx_batch_entropy(X)
        assert batch_entropy.shape == ()
