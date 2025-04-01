"""Test the sklearn models"""

# pylint: disable=attribute-defined-outside-init
import pytest

from alien.models import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    Output,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tests.conftest import ENSEMBLE_SIZE, N_CLASSES, SEED, N


class TestGaussianProcessRegressor:
    """Test the Gaussian Process wrapper"""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = GaussianProcessRegressor(X=self.X, y=self.y, random_seed=SEED)
        self.regressor.fit(X=self.X, y=self.y)

    def test_fit(self):
        """Test the fit method."""
        regressor = self.regressor
        regressor.fit(X=self.X, y=self.y)

    def test_predict(self):
        """Test the predict method."""
        X, y, regressor = self.X, self.y, self.regressor
        regressor.fit(X=X, y=y)
        X_pred = X[:N]
        y_pred = regressor.predict(X_pred)
        assert y_pred.shape == (X_pred.shape[0],)

    def test_covariance(self):
        """Test the covariance method."""
        X, y, regressor = self.X, self.y, self.regressor
        regressor.fit(X=X, y=y)
        X_pred = X[:N]
        cov = regressor.covariance(X_pred)
        assert cov.shape == (X_pred.shape[0], X_pred.shape[0])


class TestRandomForestRegressor:
    """Test the RandomForestRegressor class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = RandomForestRegressor(X=self.X, y=self.y, ensemble_size=ENSEMBLE_SIZE)
        self.regressor.fit()

    def test_predict(self):
        """Test the predict method."""
        preds = self.regressor.predict(self.X)
        assert preds.shape == (self.X.shape[0], 1)

    def test_ensemble(self):
        """Test the predict_samples method."""
        ensemble_preds = self.regressor.predict_samples(self.X, n=N)
        assert ensemble_preds.shape == (self.X.shape[0], N, 1)


class TestExtraTreesRegressor:
    """Test the ExtraTreesRegressor class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = ExtraTreesRegressor(X=self.X, y=self.y, ensemble_size=ENSEMBLE_SIZE)
        self.regressor.fit()

    def test_predict(self):
        """Test the predict method."""
        preds = self.regressor.predict(self.X)
        assert preds.shape == (self.X.shape[0], 1)

    def test_ensemble(self):
        """Test the predict_samples method."""
        ensemble_preds = self.regressor.predict_samples(self.X, n=N)
        assert ensemble_preds.shape == (self.X.shape[0], N, 1)


class TestGradientBoostingRegressor:
    """Test the GradientBoostingRegressor class"""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = GradientBoostingRegressor(X=self.X, y=self.y, ensemble_size=ENSEMBLE_SIZE)
        self.regressor.fit()

    def test_predict(self):
        """Test the predict method."""
        preds = self.regressor.predict(self.X)
        assert preds.shape == (self.X.shape[0], 1)

    def test_ensemble(self):
        """Test the predict_samples method."""
        ensemble_preds = self.regressor.predict_samples(self.X, n=N)
        assert ensemble_preds.shape == (self.X.shape[0], N, 1)


class TestHistGradientBoostingRegressor:
    """Test the HistGradientBoostingRegressor class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.regressor = HistGradientBoostingRegressor(X=self.X, y=self.y, ensemble_size=ENSEMBLE_SIZE)
        self.regressor.fit()

    def test_predict(self):
        """Test the predict method."""
        preds = self.regressor.predict(self.X)
        assert preds.shape == (self.X.shape[0], 1)

    def test_ensemble(self):
        """Test the predict_samples method."""
        ensemble_preds = self.regressor.predict_samples(self.X, n=N)
        assert ensemble_preds.shape == (self.X.shape[0], N, 1)


class TestRandomForestClassifier:
    """Test the RandomForestClassifier class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class):
        """Setup pytest fixture."""
        self.X = get_X
        self.y_class = get_y_class

    def test_predict_logit(self):
        """Test the predict method."""
        classifier = RandomForestClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.LOGIT)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_prob(self):
        """Test the predict method."""
        classifier = RandomForestClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() <= 1
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_class(self):
        """Test the predict method."""
        classifier = RandomForestClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.CLASS)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, 1)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() < N_CLASSES
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, 1)


class TestExtraTreesClassifier:
    """Test the ExtraTreesClassifier class."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class):
        """Setup pytest fixture."""
        self.X = get_X
        self.y_class = get_y_class

    def test_predict_logit(self):
        """Test predict method for logit output."""
        classifier = ExtraTreesClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.LOGIT)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_prob(self):
        """Test the predict method for prob output."""
        classifier = ExtraTreesClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() <= 1
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_class(self):
        """Test the predict method for class output."""
        classifier = ExtraTreesClassifier(X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.CLASS)
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, 1)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() < N_CLASSES
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, 1)


class TestHistGradientBoostingClassifier:
    """Test the HistGradientBoostingClassifier"""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class):
        """Setup pytest fixture."""
        self.X = get_X
        self.y_class = get_y_class

    def test_predict_prob(self):
        """Test the predict method for probability output."""
        classifier = HistGradientBoostingClassifier(
            X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB
        )
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() <= 1
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_class(self):
        """Test the predict method for class output."""
        classifier = HistGradientBoostingClassifier(
            X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.CLASS
        )
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, 1)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() < N_CLASSES
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, 1)


class TestGradientBoostingClassifier:
    """Test the GradientBoostingClassifier."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y_class):
        """Setup pytest fixture."""
        self.X = get_X
        self.y_class = get_y_class

    def test_predict_prob(self):
        """Test the predict method for probability output."""
        classifier = GradientBoostingClassifier(
            X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.PROB
        )
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, N_CLASSES)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() <= 1
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, N_CLASSES)

    def test_predict_class(self):
        """Test the predict method for class output."""
        classifier = GradientBoostingClassifier(
            X=self.X, y=self.y_class, ensemble_size=ENSEMBLE_SIZE, output=Output.CLASS
        )
        classifier.fit()
        ensemble_preds = classifier.predict_fixed_ensemble(self.X)
        assert ensemble_preds.shape == (self.X.shape[0], ENSEMBLE_SIZE, 1)
        assert ensemble_preds.min() >= 0 and ensemble_preds.max() < N_CLASSES
        preds = classifier.predict_samples(self.X)
        assert preds.shape == (self.X.shape[0], 1, 1)
