"""Test CatBoost models."""

# pylint: disable=attribute-defined-outside-init
import pytest

from alien.models import CatBoostClassifier, CatBoostRegressor
from alien.models.models import Output
from tests.conftest import ENSEMBLE_SIZE, N_CLASSES, SEED, VIRTUAL_ENSEMBLES, N_TREES


class TestCatBoostRegressor:
    """Test CatBoost regressor."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = CatBoostRegressor(X=self.X, y=self.y, random_seed=SEED, ensemble_size=ENSEMBLE_SIZE, iterations=N_TREES)

    def test_fit(self):
        """Test fit method."""
        regressor = self.regressor
        regressor.fit(X=self.X, y=self.y, random_seed=SEED)
        ensemble_preds = regressor.predict_fixed_ensemble(self.X)
        expected_shape = (self.X.shape[0], ENSEMBLE_SIZE)
        assert (
            ensemble_preds.shape == expected_shape
        ), f"Ensemble prediction expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_not_implemented(self):
        # pylint: disable=protected-access
        regressor = self.regressor
        regressor.fit(X=self.X, y=self.y, random_seed=SEED)
        with pytest.raises(NameError):
            _ = regressor.predict(self.X)
            _ = regressor._forward(self.X)
            _ = regressor._prepare_batch(self.X)

    def test_load_save(self, tmp_path):
        regressor = self.regressor
        regressor.fit(X=self.X, y=self.y, random_seed=SEED)
        regressor.save(str(tmp_path / "catboost_model"))
        loaded = CatBoostRegressor.load(str(tmp_path / "catboost_model"))
        assert isinstance(loaded, CatBoostRegressor)


class TestCatboostClassifier:
    """Test CatBoost classifier."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y_class
        self.y_class = get_y_class

    def test_predict_logit(self):
        """Test predict method for logit output."""
        X, y = self.X, self.y
        classifier = CatBoostClassifier(X=X, y=y, ensemble_size=ENSEMBLE_SIZE, output=Output.LOGIT, iterations=N_TREES)
        classifier.fit()
        preds = classifier.predict(X)
        expected_shape = (X.shape[0], N_CLASSES)
        assert preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(
            X, virtual_ensemble_count=VIRTUAL_ENSEMBLES, prediction_type="Logit"
        )
        expected_shape = (X.shape[0], ENSEMBLE_SIZE * VIRTUAL_ENSEMBLES, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_fit_predict_prob(self):
        """Test fit and predict method for probability."""
        X, y = self.X, self.y
        classifier = CatBoostClassifier(
            X=X, y=y, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB, iterations=N_TREES, n_jobs=2
        )
        classifier.fit()
        preds = classifier.predict(X)
        expected_shape = (X.shape[0], N_CLASSES)
        assert preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(
            X, virtual_ensemble_count=VIRTUAL_ENSEMBLES, prediction_type="Prob"
        )
        expected_shape = (X.shape[0], ENSEMBLE_SIZE * VIRTUAL_ENSEMBLES, N_CLASSES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_fit_predict_class(self):
        """Test fit and predict method for class output."""
        X, y = self.X, self.y
        classifier = CatBoostClassifier(X=X, y=y, ensemble_size=ENSEMBLE_SIZE, output=Output.CLASS, iterations=N_TREES)
        classifier.fit()
        preds = classifier.predict(X)
        expected_shape = (X.shape[0],)
        assert preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(X)
        expected_shape = (X.shape[0], ENSEMBLE_SIZE, 1)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
        ensemble_preds = classifier.predict_fixed_ensemble(
            X, virtual_ensemble_count=VIRTUAL_ENSEMBLES, prediction_type="Class"
        )
        expected_shape = (X.shape[0], ENSEMBLE_SIZE * VIRTUAL_ENSEMBLES)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
