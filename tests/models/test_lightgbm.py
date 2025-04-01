"""Tests for the LightGBM wrapper."""

# pylint: disable=attribute-defined-outside-init
import pytest
from pytest import importorskip

from alien.models import LightGBMClassifier, LightGBMRegressor
from alien.models.lightgbm import LR_DEFAULT
from alien.models.models import Output
from tests.conftest import ENSEMBLE_SIZE, N_CLASSES

lgb = importorskip("lightgbm")


class TestLightGBMRegressor:
    """Test the LightGBM regressor wrapper"""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_data):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.data = get_data
        self.regressor = LightGBMRegressor(data=self.data, ensemble_size=ENSEMBLE_SIZE)

    def test_fit(self):
        """Test the fit method."""
        self.regressor.fit(X=self.X, y=self.y)

    def test_predict(self):
        """Test the predict method."""
        X, y = self.X, self.y
        self.regressor.fit(X=X, y=y)
        pred = self.regressor.predict(X[:10])
        assert pred.shape == (10,)

    # TODO: not yet implemented
    # def test_covariance(self):
    #     TestLightGBMRegressor.regressor.fit(X=X, y=y)
    #     cov = TestLightGBMRegressor.regressor.covariance(X[:10])
    #     assert cov.shape == (10, 10)

    def test_learning_rate(self):
        """Test setting the learning rate."""
        regressor = self.regressor
        lr_1 = regressor.learning_rate
        assert lr_1 == LR_DEFAULT
        lr_2 = LR_DEFAULT + 0.05
        regressor.learning_rate = lr_2
        assert regressor.learning_rate == lr_2


class TestLightGBMClassifier:
    """Test the LightGBM classifier wrapper"""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class, get_data):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y_class
        self.data = get_data

    def test_fit_predict_logit(self):
        """Test predict method for logit output."""
        X, y = self.X, self.y
        classifier = LightGBMClassifier(X=X, y=y, ensemble_size=ENSEMBLE_SIZE, output=Output.LOGIT)
        classifier.fit()
        preds = classifier.predict(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_fit_predict_prob(self):
        """Test fit and predict method for probability."""
        X, y = self.X, self.y
        classifier = LightGBMClassifier(X=X, y=y, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB)
        classifier.fit()
        preds = classifier.predict(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
