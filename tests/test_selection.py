# pylint disable=attribute-defined-outside-init
import numpy as np
import pytest
from pytest import importorskip

from alien.data import TeachableDataset
from alien.models import PytorchRegressor
from alien.selection import (
    BAITSelector,
    CovarianceSelector,
    EntropySelector,
    ExpectedImprovementSelector,
    KmeansSelector,
    RandomSelector,
    ThompsonSelector,
    TimestampSelector,
)
from tests.conftest import BATCH_SIZE, ENSEMBLE_SIZE, N_SAMPLES

# Skip testing selectors if torch is not installed
torch = importorskip("torch")


class TestRandomSelector:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_nn_model
        self.regressor = PytorchRegressor(X=self.X, y=self.y, model=self.model, uncertainty="dropout")
        self.selector = RandomSelector(model=self.regressor, samples=self.X)

    def test_selection(self):
        selected = self.selector.select(batch_size=BATCH_SIZE)
        expected_shape = (BATCH_SIZE, self.X.shape[1])
        assert selected.shape == expected_shape

    def test_prediction_prior(self):
        samples_torch = torch.from_numpy(self.X).type(torch.float32)
        priors = self.selector.prediction_prior(samples_torch)
        assert (
            priors.shape == (len(samples_torch),)
            if isinstance(priors, np.ndarray)
            else priors.shape == (len(samples_torch), 1)
        ), "Prior should be of same size as samples."
        assert (priors <= 1).all() and (priors >= 0).all(), "Priors should be between 0 and 1."

    def test_model_predict(self):
        samples_torch = torch.from_numpy(self.X).type(torch.float32)

        _ = self.selector.model_predict(samples_torch)
        assert samples_torch is self.selector._last_X, "Tensor should be saved as last input"
        predicted = self.selector.model_predict(samples_torch + 1)
        assert samples_torch is not self.selector._last_X, "_last_X should change after a new prediction"
        assert predicted is self.selector._last_pred


class TestTimestampSelector:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        data = {"X": self.X, "y": self.y, "t": np.arange(N_SAMPLES)[::-1]}
        self.db = TeachableDataset.from_data(data)
        self.selector = TimestampSelector(samples=self.db)

    def test_selection(self):
        selected = self.selector.select(batch_size=BATCH_SIZE)
        assert (selected[::-1, "X"].data == self.db[-BATCH_SIZE:, "X"].data).all()


class TestCovarianceSelector:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_nn_model
        self.regressor = PytorchRegressor(X=self.X, y=self.y, model=self.model, uncertainty="dropout")
        self.selector = CovarianceSelector(model=self.regressor, samples=self.X, batch_size=BATCH_SIZE, n_rounds=100)

    def test_selection(self):
        samples_torch = torch.from_numpy(self.X).type(torch.float32)
        selected = self.selector.select(samples=samples_torch, batch_size=BATCH_SIZE, verbose=True)
        expected_shape = (BATCH_SIZE, self.X.shape[1])
        assert selected.shape == expected_shape


class TestThompsonSelector:
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_nn_model
        self.regressor = PytorchRegressor(X=self.X, y=self.y, model=self.model, uncertainty="dropout")
        self.selector = ThompsonSelector(model=self.regressor, samples=self.X, batch_size=BATCH_SIZE, sign=-1)

    def test_selection(self):
        samples_torch = torch.from_numpy(self.X).type(torch.float32)
        selected = self.selector.select(samples=samples_torch, batch_size=BATCH_SIZE, verbose=True)
        expected_shape = BATCH_SIZE, self.X.shape[1]
        assert (
            (selected.shape == expected_shape)
            if isinstance(selected, np.ndarray)
            else (selected.shape == (BATCH_SIZE, 1, self.X.shape[1]))
        ), f"Expected shape {expected_shape} for selection. Got {selected.shape}"


class TestKmeansSelector:
    # pylint: disable=too-few-public-methods
    @pytest.fixture(autouse=True)
    def setup(self, get_X, get_y, get_nn_model):
        """Setup pytest fixture."""
        self.X = get_X
        self.y = get_y
        self.model = get_nn_model
        self.regressor = PytorchRegressor(X=self.X, y=self.y, model=self.model, uncertainty="dropout")
        self.selector = KmeansSelector(model=self.regressor)

    def test_selection(self):
        selector = self.selector
        selected = selector.select(samples=self.X, batch_size=BATCH_SIZE)
        expected_shape = (BATCH_SIZE, self.X.shape[1])
        assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"


class TestEntropySelector:
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
        self.selector = EntropySelector(model=self.regressor, samples=self.X_tensor, batch_size=BATCH_SIZE)
        self.lazy_selector = EntropySelector(
            model=self.regressor, samples=self.X_tensor, batch_size=BATCH_SIZE, precompute_entropy=False, buffer_size=10
        )

    def test_selection(self):
        selector = self.selector
        selected = selector.select(samples=self.X_tensor, batch_size=BATCH_SIZE, verbose=True)
        expected_shape = (BATCH_SIZE, self.X_tensor.shape[1])
        assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"

        lazy_selected = self.lazy_selector.select(samples=self.X_tensor, batch_size=BATCH_SIZE, verbose=True)
        assert (
            lazy_selected.shape == expected_shape
        ), f"Expected shape {expected_shape} for lazy selection. Got {lazy_selected.shape}"


class TestBAITSelector:
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
        self.selector = BAITSelector(model=self.regressor, samples=self.X_tensor, batch_size=BATCH_SIZE)

    def test_selection(self):
        selector = self.selector
        selected = selector.select(
            samples=self.X_tensor[:30], labelled_samples=self.X_tensor[30:], batch_size=BATCH_SIZE, verbose=True
        )
        expected_shape = (BATCH_SIZE, self.X_tensor.shape[1])
        assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"
        selected = selector.select(samples=self.X_tensor[:30], batch_size=BATCH_SIZE, verbose=True)
        assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"


class TestExpectedImprovementSelector:
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
        self.selector = ExpectedImprovementSelector(model=self.regressor, samples=self.X_tensor, batch_size=BATCH_SIZE)

    # def test_selection(self):
    #     selector = self.selector
    #     selected = selector.select(
    #         samples=self.X_tensor[:30],
    #         labelled_samples=self.X_tensor[30:],
    #         y_labelled=self.y_tensor[30:],
    #         batch_size=BATCH_SIZE,
    #         verbose=True,
    #     )
    #     expected_shape = (BATCH_SIZE, self.X_tensor.shape[1])
    #     assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"


def test_optimize_batch():
    # TODO
    pass
