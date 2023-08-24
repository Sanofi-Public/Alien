import numpy as np
from pytest import importorskip

from alien.data import TeachableDataset
from alien.models import PytorchRegressor
from alien.selection import (
    CovarianceSelector,
    KmeansSelector,
    RandomSelector,
    TimestampSelector,
    ThompsonSelector,
)

# Skip testing selectors if torch is not installed
torch = importorskip("torch")
nn = torch.nn

# TODO: test optimize_batch

seed = 0
np.random.seed(seed)
n_samples = 64
n_features = 16
X = np.random.normal(size=(n_samples, n_features))
y = np.random.normal(size=n_samples)
batch_size = 10


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class TestRandomSelector:
    model = NeuralNetwork()
    regressor = PytorchRegressor(X=X, y=y, model=model, uncertainty='dropout')
    selector = RandomSelector(model=regressor, samples=X)

    def test_selection(self):
        selected = TestRandomSelector.selector.select(batch_size=batch_size)
        expected_shape = (batch_size, X.shape[1])
        assert selected.shape == expected_shape

    def test_prediction_prior(self):
        samples_torch = torch.from_numpy(X).type(torch.float32)

        priors = TestRandomSelector.selector.prediction_prior(samples_torch)
        assert priors.shape == (len(samples_torch),), "Prior should be of same size as samples."

        assert (priors <= 1).all() and (priors >= 0).all(), "Priors should be between 0 and 1."

    def test_model_predict(self):
        samples_torch = torch.from_numpy(X).type(torch.float32)

        _ = TestRandomSelector.selector.model_predict(samples_torch)
        assert (
            samples_torch is TestRandomSelector.selector._last_X
        ), "Tensor should be saved as last input"
        predicted = TestRandomSelector.selector.model_predict(samples_torch + 1)
        assert (
            samples_torch is not TestRandomSelector.selector._last_X
        ), "_last_X should change after a new prediction"
        assert predicted is TestRandomSelector.selector._last_pred


class TestTimestampSelector:
    data = {"X": X, "y": y, "t": np.arange(n_samples)[::-1]}
    db = TeachableDataset.from_data(data)
    selector = TimestampSelector(samples=db)

    def test_selection(self):
        selected = TestTimestampSelector.selector.select(batch_size=batch_size)
        assert (selected[::-1, "X"].data == TestTimestampSelector.db[-batch_size:, "X"].data).all()


class TestCovarianceSelector:
    model = NeuralNetwork()
    regressor = PytorchRegressor(X=X, y=y, model=model, uncertainty='dropout')
    selector = CovarianceSelector(
        model=regressor, samples=X, batch_size=batch_size, n_rounds=100, similarity=0.1
    )

    def test_selection(self):
        samples_torch = torch.from_numpy(X).type(torch.float32)
        selected = TestCovarianceSelector.selector.select(
            samples=samples_torch, batch_size=batch_size, verbose=True
        )
        expected_shape = (batch_size, X.shape[1])
        assert selected.shape == expected_shape

class TestThompsonSelector:
    model = NeuralNetwork()
    regressor = PytorchRegressor(X=X, y=y, model=model, uncertainty='dropout')
    selector = ThompsonSelector(
        model=regressor, samples=X, batch_size=batch_size, sign=-1
    )

    def test_selection(self):
        samples_torch = torch.from_numpy(X).type(torch.float32)
        selected = TestThompsonSelector.selector.select(
            samples=samples_torch, batch_size=batch_size, verbose=True
        )
        expected_shape = batch_size, X.shape[1]
        assert selected.shape == expected_shape, f"Expected shape {expected_shape} for selection. Got {selected.shape}"




class TestKmeansSelector:
    # pylint: disable=too-few-public-methods
    model = NeuralNetwork()
    regressor = PytorchRegressor(model=model, X=X, y=y, uncertainty='dropout')
    selector = KmeansSelector(model=regressor)

    def test_selection(self):
        selector = TestKmeansSelector.selector
        selected = selector.select(samples=X, batch_size=batch_size)
        expected_shape = (batch_size, X.shape[1])
        assert (
            selected.shape == expected_shape
        ), f"Expected shape {expected_shape} for selection. Got {selected.shape}"


def test_optimize_batch():
    # TODO
    pass
