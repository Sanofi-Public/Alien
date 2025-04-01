"""Test DeepChem regressors."""

# pylint: disable=attribute-defined-outside-init
import pytest
from pytest import importorskip

from alien.data import DeepChemDataset
from alien.models import DeepChemRegressor
from tests.conftest import DEEPCHEM_DATA_PATH, ENSEMBLE_SIZE

dc = importorskip("deepchem")
torch = importorskip("torch")
keras = importorskip("keras")
class TestDeepChemMCDropoutRegressorTorch:
    """Test DeepChemRegressor with PyTorch model."""
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_dc_torch_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_dc_torch_model
        self.data = DeepChemDataset.from_csv(DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="molgraphconv")
        self.data.y.astype("float32")
        self.regressor = DeepChemRegressor(  # should return DeepChemPytorchRegressor
            model=self.model, data=self.data, ensemble_size=ENSEMBLE_SIZE, uncertainty="dropout"
        )
    def test_fit(self):
        """Test fit method."""
        regressor = self.regressor
        regressor.fit()
    def test_predict(self):
        """Test predict method."""
        regressor = self.regressor
        data = self.data
        preds = regressor.predict(data.X)
        assert preds.shape == (data.shape[0], 1)
    def test_covariance(self):
        """Test covariance method."""
        regressor = self.regressor
        data = self.data
        cov = regressor.covariance(data.X)
        assert cov.shape == (data.shape[0], data.shape[0], 1, 1)
    def test_get_train_dataset(self):
        """Test get_train_dataset method."""
        # pylint: disable=protected-access
        regressor = self.regressor
        train_np = regressor.get_train_dataset(X=self.X, y=self.y)
        assert isinstance(train_np, type(self.data._to_DC()))
        train_alien = regressor.get_train_dataset(X=self.data)
        assert isinstance(train_alien, type(self.data._to_DC()))
class TestDeepChemMCDropoutRegressorKeras:
    """Test DeepChemRegressor with Keras model."""
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_dc_keras_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_dc_keras_model
        self.data = DeepChemDataset.from_csv(DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="convmol")
        nodropout = self.model.model.graph_convs[-1]
        self.regressor = DeepChemRegressor(  # should return DeepChemKerasRegressor
            self.model,
            self.data,
            ensemble_size=ENSEMBLE_SIZE,
            frozen_layers=nodropout,
            uncertainty="dropout",
        )
    def test_fit(self):
        """Test fit method."""
        self.regressor.fit()
    def test_predict(self):
        """Test predict method."""
        regressor = self.regressor
        data = self.data
        preds = regressor.predict(data)
        assert preds.shape == (data.shape[0], 1)
    def test_ensemble(self):
        """Test ensemble prediction."""
        regressor = self.regressor
        data = self.data
        ensemble_preds = regressor.predict_fixed_ensemble(data)
        expected_shape = (data.shape[0], ENSEMBLE_SIZE)
        assert ensemble_preds.shape == expected_shape, f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"
    def test_covariance(self):
        """Test covariance method."""
        regressor = self.regressor
        data = self.data
        cov = regressor.covariance(data)
        assert cov.shape == (data.shape[0], data.shape[0])
    def test_get_train_dataset(self):
        """Test get_train_dataset method."""
        # pylint: disable=protected-access
        regressor = self.regressor
        train_np = regressor.get_train_dataset(X=self.X, y=self.y)
        assert isinstance(train_np, type(self.data._to_DC()))
        train_alien = regressor.get_train_dataset(X=self.data)
        assert isinstance(train_alien, type(self.data._to_DC()))
class TestDeepChemKerasLaplaceRegressor:
    """Test DeepChemRegressor with Keras model and Laplace uncertainty."""
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_dc_keras_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_dc_keras_model
        self.data = DeepChemDataset.from_csv(DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="convmol")
        nodropout = self.model.model.graph_convs[-1]
        self.regressor = DeepChemRegressor(  # should return DeepChemKerasRegressor
            self.model,
            self.data,
            ensemble_size=ENSEMBLE_SIZE,
            frozen_layers=nodropout,
            uncertainty="laplace",
        )
    def test_fit(self):
        """Test fit method."""
        regressor = self.regressor
        # regressor.fit(epochs=100, val_data=data)
        regressor.fit(epochs=100)
    def test_predict(self):
        """Test predict method."""
        regressor = self.regressor
        data = self.data
        preds = regressor.predict(data)
        assert preds.shape == (data.shape[0], 1)
    def test_covariance(self):
        """Test covariance method."""
        regressor = self.regressor
        data = self.data
        self.regressor.fit(epochs=10)  # Must fit to get covariance
        cov = regressor.covariance(data)
        assert cov.shape == (data.shape[0], data.shape[0])
