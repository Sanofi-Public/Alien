"""Module for testing Pytorch models."""

# pylint: disable=attribute-defined-outside-init
import pytest
from pytest import importorskip

from alien.models import PytorchRegressor, Regressor
from alien.models.utils import get_base_model
from tests.conftest import ENSEMBLE_SIZE

torch = importorskip("torch")


class TestPytorchMCDropoutRegressor:
    """Test PytorchRegressor with dropout uncertainty."""

    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        """Setup pytest fixture."""
        self.X_tensor = torch.from_numpy(get_X).type(torch.float32)
        self.y_tensor = torch.from_numpy(get_y).type(torch.float32)
        self.model = get_nn_model
        self.regressor = PytorchRegressor(
            model=self.model,
            X=self.X_tensor,
            y=self.y_tensor,
            ensemble_size=ENSEMBLE_SIZE,
            frozen_layers=self.model.linear_relu_stack[-1],
            uncertainty="dropout",
        )

    def test_fit(self):
        """Test fit method."""
        regressor = self.regressor
        regressor.fit(
            X=self.X_tensor,
            y=self.y_tensor,
            batch_size=16,
        )

    def test_covariance(self):
        """Test covariance method."""
        regressor = self.regressor
        cov_mat = regressor.covariance(self.X_tensor)
        expected_shape = (self.X_tensor.shape[0], self.X_tensor.shape[0])
        assert cov_mat.shape[:2] == expected_shape, "Shape of output is correct"

    def test_ensemble(self):
        """Test ensemble prediction."""
        regressor = self.regressor
        ensemble_preds = regressor.predict_fixed_ensemble(self.X_tensor)
        expected_shape = (
            self.X_tensor.shape[0],
            ENSEMBLE_SIZE,
        )
        assert ensemble_preds.shape[:2] == expected_shape, "Shape of the output is correct"

    def test_creation(self):
        """Test creation of PytorchRegressor."""
        # pylint: disable=abstract-class-instantiated
        regressor = Regressor(
            model=self.model, X=self.X_tensor, y=self.y_tensor, ensemble_size=ENSEMBLE_SIZE, uncertainty="dropout"
        )
        assert isinstance(regressor, PytorchRegressor)


class TestPytorchLastLayerLaplaceRegressor:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        self.X_tensor = torch.from_numpy(get_X).type(torch.float32)
        self.y_tensor = torch.from_numpy(get_y).type(torch.float32)
        self.model = get_nn_model
        self.regressor = PytorchRegressor(
            model=self.model,
            X=self.X_tensor,
            y=self.y_tensor,
            uncertainty="laplace",
        )

    def test_find_last_layer(self):
        regressor = self.regressor
        _ = regressor.find_last_layer(self.X_tensor)
        assert isinstance(
            regressor.last_layer, torch.nn.Linear
        ), f"Last layer should be nn.Linear, not {type(regressor.last_layer)}"

    def test_embeddings(self):
        regressor = self.regressor
        _, embd_1 = regressor.predict_with_embedding(self.X_tensor)
        embd_2 = regressor.embedding(self.X_tensor)
        assert (embd_1 == embd_2).all(), "embedding(*) and predict_with_embedding(*) should return the same embeddings."

    def test_predict(self):
        regressor = self.regressor
        preds_1, _ = regressor.predict_with_embedding(self.X_tensor)
        preds_2 = regressor.predict(self.X_tensor)
        assert (preds_1 == preds_2).all(), "predict (*) and predict_with_embedding(*) should return the same output."


class TestLaplaceRegressor:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        self.X, self.y = get_X, get_y
        self.model = get_nn_model

    def test_creation(self):
        # pylint: disable=abstract-class-instantiated
        regressor = Regressor(
            model=self.model,
            X=self.X,
            y=self.y,
            uncertainty="laplace",
        )
        assert isinstance(regressor, PytorchRegressor)


def test_get_base_model(get_nn_model):
    # pylint: disable=abstract-class-instantiated
    pt_model = get_nn_model
    al_model = Regressor(pt_model)
    assert get_base_model(al_model) == pt_model
    assert get_base_model(al_model, framework="torch") == pt_model
    with pytest.raises(ValueError):
        assert get_base_model(al_model, framework="keras")
