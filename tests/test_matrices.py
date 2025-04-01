import numpy as np

from alien.matrices import EnsembleMatrix
from tests.conftest import ENSEMBLE_SIZE, N_FEATURES, N_SAMPLES


def test_matrices():
    ensemble_preds = np.random.normal(size=(N_SAMPLES, ENSEMBLE_SIZE))
    matrix = EnsembleMatrix(ensemble_preds)
    # Matrix becomes square
    assert matrix.shape == (N_SAMPLES, N_SAMPLES)
