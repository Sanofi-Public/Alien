import numpy as np

from alien.stats import (
    augment_ensemble,
    covariance_from_ensemble,
    covariance_from_similarity,
    ensemble_from_covariance,
    entropy,
    group_edges,
    group_sizes,
    multiply_cov,
    similarity_cosine,
    similarity_exp,
    similarity_gaussian,
    std_dev_from_ensemble,
    variance_from_ensemble,
)
from tests.conftest import ENSEMBLE_SIZE, N_CLASSES, N_SAMPLES

multiple = 0.73


def test_multiply_cov(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    std_1 = np.std(ensemble_preds, axis=-1)
    ensemble_2 = multiply_cov(ensemble_preds, multiple)
    std_2 = np.std(ensemble_2, axis=-1)
    assert ensemble_2.shape == ensemble_preds.shape
    assert np.isclose(multiple * std_1, std_2).all()
    ensemble_3 = multiply_cov(ensemble_preds, multiple=1)
    assert np.isclose(ensemble_3, ensemble_preds).all()


def test_covariance_from_ensemble(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    cov = covariance_from_ensemble(ensemble_preds)
    assert cov.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])
    ensemble_preds_list = list(ensemble_preds)
    cov = covariance_from_ensemble(ensemble_preds_list)
    assert cov.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_covariance_from_similarity(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    sim = np.corrcoef(ensemble_preds)
    var = np.var(ensemble_preds, axis=-1)
    cov = covariance_from_similarity(sim, var)
    assert cov.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_cosine(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    sim = similarity_cosine(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_exp(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    sim = similarity_exp(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_gaussian(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    sim = similarity_gaussian(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_ensemble_from_covariance(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    cov = covariance_from_ensemble(ensemble_preds)
    mean = np.mean(ensemble_preds, axis=-1)
    preds_2 = ensemble_from_covariance(mean, cov, ensemble_size=ENSEMBLE_SIZE, epsilon=1e-6)
    assert preds_2.shape == ensemble_preds.shape
    
    
def test_augment_ensemble(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    N = 19
    ensemble_2 = augment_ensemble(ensemble_preds, N=N)
    assert ensemble_2.shape == (N_SAMPLES, N)
    N = ENSEMBLE_SIZE - 2
    ensemble_2 = augment_ensemble(ensemble_preds, N=N)
    assert ensemble_2.shape == ensemble_preds.shape


def test_variance_from_ensemble(get_ensemble_preds):
    ensemble_preds = get_ensemble_preds
    var = variance_from_ensemble(ensemble_preds)
    assert var.shape == (ensemble_preds.shape[0],)
    ensemble_preds_list = list(ensemble_preds)
    var = variance_from_ensemble(ensemble_preds_list, pbar=True)
    assert var.shape == (ensemble_preds.shape[0],)


# def test_std_dev_from_ensemble(get_ensemble_preds):
#     ensemble_preds = get_ensemble_preds
#     std_dev = std_dev_from_ensemble(ensemble_preds, verbose=True)
#     assert std_dev.shape == (ensemble_preds.shape[0],)
#     ensemble_preds_list = list(ensemble_preds)
#     std_dev = std_dev_from_ensemble(ensemble_preds_list, verbose=True)
#     assert std_dev.shape == (ensemble_preds.shape[0],)


# def test_entropy(get_y_class):
#     y_class = get_y_class
#     entropy_ = entropy(y_class)
#     assert entropy_.shape == y_class.shape
#     y_class_prob = np.zeros((y_class.shape[0], N_CLASSES))
#     y_class_prob[np.arange(y_class.shape[0]), y_class] = 1
#     entropy_ = entropy(y_class_prob)
#     assert entropy_.shape == y_class


def test_group_sizes():
    size = 7
    group_size = 1
    for group_size in range(1, size + 1):
        out = group_sizes(size, group_size)
        assert len(out) == int(np.ceil(size / group_size))
        assert sum(out) == size
        assert max(out) - min(out) <= 1


def test_group_edges():
    size = 7
    group_size = 1
    for group_size in range(1, size + 1):
        out = group_edges(size, group_size)
        assert len(out) == int(np.ceil(size / group_size))
