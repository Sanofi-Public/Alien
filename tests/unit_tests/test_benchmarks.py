import os

import numpy as np
from pytest import importorskip
from scipy.stats import sem as sp_sem

from alien.benchmarks import (
    KL_divergence,
    Scatter,
    Score,
    best_multiple,
    run_experiments,
)
from alien.benchmarks.metrics import sem
from alien.models import PytorchRegressor

torch = importorskip("torch")
nn = torch.nn

np.random.seed(0)
iters = list(range(10))
y = np.random.normal(size=len(iters))


class NeuralNetwork(nn.Module):
    """Sample Pytorch Neural Network class"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 32),
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


class TestScore:
    score = Score(x=iters, y=y, err=None, name="test_score")

    def test_save(self, tmp_path):
        d = tmp_path / "scores"
        d.mkdir()
        file_path = str(d / "score_1.pkl")
        score = TestScore.score
        score.save(file_path=file_path)
        assert os.path.isfile(file_path)

    def test_load(self, tmp_path):
        d = tmp_path / "scores"
        d.mkdir()
        file_path = str(d / "score_1.pkl")
        score = TestScore.score
        score.save(file_path=file_path)
        score_loaded = Score.load(file_path)
        assert (score_loaded.x.data == score.x.data).all()

    def test_append(self):
        score = TestScore.score
        score.append(len(iters), -0.5)
        assert len(score.x) == len(iters) + 1

    def test_average_runs(self):
        score = TestScore.score
        _ = Score.average_runs(score, score, length="longest")
        _ = Score.average_runs(score, score, length="max")


def test_kl_divergence():
    preds = y + np.random.normal(size=len(y))
    std_devs = np.random.normal(size=len(y))
    kl_div = KL_divergence(preds, std_devs, y, normalize=True)
    assert isinstance(kl_div, float)


def test_best_multiple():
    preds = y + np.random.normal(size=len(y))
    std_devs = np.random.normal(size=len(y))
    _ = best_multiple(preds, std_devs, y)


def test_sem():
    ensemble_preds = np.random.normal(size=(len(iters), 13))
    test_sem = sem(ensemble_preds)
    actual_sem = sp_sem(ensemble_preds, ddof=0)
    assert test_sem.shape == actual_sem.shape
    assert np.isclose(test_sem, actual_sem).all()


def test_run_experiments(tmp_path):
    experiment_path = tmp_path / "experiment_output"
    experiment_path.mkdir(exist_ok=True)
    experiment_path = str(experiment_path)
    n_runs = 2

    SEED = 0
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    def fn(X, noise=True):
        return (
            np.sin(np.pi * 4 * X[..., 0])
            + np.cos(np.pi * 4 * X[..., 1])
            + (rng.uniform(-0.1, 0.1, X.shape[:-1]) if noise else 0)
        )

    def cut(X):
        return X[..., 0] ** 2 + 2 * X[..., 1] ** 2

    ensemble_size = 13
    lin = np.linspace(0, 1, 50)
    X_test = np.stack(np.meshgrid(lin, lin), axis=2)
    X = rng.uniform(0, 1, size=(1000, 2))
    X_train = X[cut(X) < 0.8]
    y_train = fn(X_train)
    model = NeuralNetwork()
    X_tensor = torch.from_numpy(X_train).type(torch.float32)
    y_tensor = torch.from_numpy(y_train).type(torch.float32)
    regressor = PytorchRegressor(
        model=model, X=X_tensor, y=y_tensor, ensemble_size=ensemble_size, uncertainty='dropout'
    )
    run_experiments(
        X_tensor,
        y_tensor,
        regressor,
        runs_dir=experiment_path,
        overwrite_old_runs=True,
        n_initial=100,
        batch_size=32,
        num_samples=float("inf"),
        selector=None,  # 'covariance', 'random', 'expected improvement'/'ei', 'greedy'
        selector_args=None,
        fit_args=None,
        n_runs=n_runs,
        ids=None,
        save_ids=True,
        random_seed=1,
        split_seed=421,
        test_size=0.2,
        timestamps=None,
        stop_samples=150,
        stop_rmse=None,
        stop_frac=None,  # suggest .85,
        # The following arguments are a bit odd, and best avoided
        peek_score=0,  # 0 if no peeking
        test_samples_x=None,
        test_samples_y=None,
    )

    for i in range(n_runs):
        run_dir = os.path.join(experiment_path, f"run_0{i}")
        assert os.path.isdir(run_dir)
        assert os.path.isfile(os.path.join(run_dir, "RMSE.pickle"))
