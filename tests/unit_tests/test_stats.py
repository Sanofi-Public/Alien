import numpy as np

from alien.stats import (
    augment_ensemble,
    covariance_from_ensemble,
    covariance_from_similarity,
    ensemble_from_covariance,
    group_edges,
    group_sizes,
    multiply_std_dev,
    similarity_cosine,
    similarity_exp,
    similarity_gaussian,
)

np.random.seed(0)
ensemble_size = 13
N_SAMPLES = 64
ensemble_preds = np.random.normal(size=(N_SAMPLES, ensemble_size))
multiple = 0.73


def test_multiply_std_dev():
    std_1 = np.std(ensemble_preds, axis=-1)
    ensemble_2 = multiply_std_dev(ensemble_preds, multiple)
    std_2 = np.std(ensemble_2, axis=-1)
    assert ensemble_2.shape == ensemble_preds.shape
    assert np.isclose(multiple * std_1, std_2).all()


def test_covariance_from_ensemble():
    cov = covariance_from_ensemble(ensemble_preds)
    assert cov.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_covariance_from_similarity():
    sim = np.corrcoef(ensemble_preds)
    var = np.var(ensemble_preds, axis=-1)
    cov = covariance_from_similarity(sim, var)
    assert cov.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_cosine():
    sim = similarity_cosine(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_exp():
    sim = similarity_exp(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_similarity_gaussian():
    sim = similarity_gaussian(ensemble_preds)
    assert sim.shape == (ensemble_preds.shape[0], ensemble_preds.shape[0])


def test_ensemble_from_covariance():
    cov = covariance_from_ensemble(ensemble_preds)
    mean = np.mean(ensemble_preds, axis=-1)
    preds_2 = ensemble_from_covariance(mean, cov, ensemble_size=ensemble_size)
    assert preds_2.shape == ensemble_preds.shape


def test_augment_ensemble():
    N = 19
    ensemble_2 = augment_ensemble(ensemble_preds, N=N)
    assert ensemble_2.shape == (N_SAMPLES, N)
    N = ensemble_size - 2
    ensemble_2 = augment_ensemble(ensemble_preds, N=N)
    assert ensemble_2.shape == ensemble_preds.shape


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
