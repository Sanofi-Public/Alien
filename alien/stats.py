from itertools import cycle

import numpy as np

from .decorators import flatten_batch, normalize_args
from .utils import shift_seed

# pylint: disable=no-value-for-parameter


@flatten_batch(bdim=-1)
def multiply_std_dev(ensemble, multiple: float):
    """
    Modifies an ensemble to change its standard deviation by
    a factor of 'multiple'.

    :param ensemble: ensembles are over the last dimension
        only.
    """
    if multiple == 1:
        return ensemble
    means = np.mean(ensemble, axis=-1, keepdims=True)
    ensemble = multiple * ensemble + (1 - multiple) * means
    return ensemble


@flatten_batch(bdim=-1)
def augment_ensemble(preds, N: int, multiple=1.0, rng=None):
    """
    Augments ensemble predictions to an ensemble of size N
    per variable.

    Extra observations are generated from a multivariate normal
    distribution to have the same covariances (within the last
    batch dimension) as existing observations

    :param multiple: returns an ensemble with 'multiple'
        times the covariance of the original.
    """
    if multiple != 1:
        preds = multiply_std_dev(preds, multiple)

    if N <= preds.shape[-1]:
        return preds

    mean = np.mean(preds, axis=-1)
    cov = covariance_from_ensemble(preds)
    aug = ensemble_from_covariance(mean, cov, N - preds.shape[-1], rng)
    return np.concatenate((preds, aug), axis=-1)


# @flatten_batch(bdim=-2)
def covariance_from_ensemble(preds, weights=None):
    # weights.shape = (3,)
    if weights is not None:
        preds = preds * weights

    if preds.ndim == 2:
        return np.cov(preds)

    # preds.shape = (N, 50, 3)

    # subtract mean, and then reshape:
    p0 = (preds - preds.mean(axis=1, keepdims=True)).reshape(len(preds), -1)
    #                       3N different means

    # p0.shape = (N, 150)

    return (p0[None, ...] * p0[:, None, ...]).sum(axis=1) / preds.shape[1]
    # output.shape (N, N)


def std_dev_from_ensemble(preds, weights=None):
    if preds.ndim == 2:
        return np.std(preds, axis=1)

    raise NotImplementedError


# @flatten_batch(bdim=-2, to_flatten=2, degree)
def ensemble_from_covariance(mean, cov, ensemble_size, rng=None, random_seed=None):
    """ """
    if rng is None:
        rng = np.random.default_rng(random_seed)
    return np.moveaxis(
        rng.multivariate_normal(mean, cov, size=ensemble_size),
        -2,
        -1,
    )


@flatten_batch(bdim=-1)
@normalize_args(euclidean=True)
def similarity_gaussian(X, relevant_features=None, normalize=True, scale=1.0):
    """
    Computes a similarity matrix from a Gaussian function of distances,
    as follows:
        Restricts X to only look at 'relevant_features'.

        If 'normalize' is True, rescales the input to have standard
        deviation 1 in each feature, then rescales by another factor
        of sqrt(X.shape[-1]).

        Then the distance matrix is put into a Gaussian of std dev
        equal to 'scale'.

    """
    if relevant_features is not None:
        X = X[..., relevant_features]

    # TODO: if normalize: ...
    distance = np.linalg.norm(X[..., :, None, :] - X[..., None, :, :], axis=-1)
    similarity = np.exp(-np.square(distance) / (2 * scale))
    return similarity


@flatten_batch(bdim=-1)
@normalize_args(euclidean=True)
def similarity_exp(X, relevant_features=None, normalize=True, scale=1.0):
    # TODO: if normalize ...
    if relevant_features is not None:
        X = X[..., relevant_features]

    distance = np.linalg.norm(X[..., :, None, :] - X[..., None, :, :], axis=-1)
    similarity = np.exp(-distance / scale)
    return similarity


@flatten_batch(bdim=-1)
@normalize_args
def similarity_cosine(X, relevant_indices=None, normalize=True):
    if relevant_indices is not None:
        X = X[relevant_indices]

    l = np.linalg.norm(X, axis=-1)
    l[l == 0] = 1
    X = (X.T / l).T
    return np.sum(X[..., None, :, :] * X[..., :, None, :], axis=-1)


def covariance_from_similarity(similarity, variance):
    """
    Given a similarity matrix, and a vector of variances,
    returns a covariance matrix, using the similarity matrix
    as correlations.
    """
    std = np.sqrt(variance)
    return std[..., :, None] * similarity * std[..., None, :]


@flatten_batch(bdim=-2)
def joint_entropy(X):
    """
    Not regularized. Is liable to underestimate the
    entropy, i.e., samples may actually be less correlated
    than they appear from this calculation, due to the
    limited size of the ensemble.

    :param X: ensemble of class predictions, of shape
        ... x batch_size x ensemble_size
    """
    entropy = np.zeros(X.shape[0])
    for i, X_b in enumerate(X):
        _, counts = np.unique(X_b, axis=-1, return_counts=True)
        probs = counts / X.shape[-1]
        entropy[i] = np.sum(-np.log2(probs) * probs)

    return entropy


@flatten_batch(bdim=-1)
def joint_entropy_regularized(X, n_classes=None, rng=None, random_seed=None):
    """
    :param X: ensemble of class predictions, of shape
        ... x batch_size x ensemble_size

    :param n_classes: to avoid computing it, you can
        pass in the number of classes.
    """
    if rng is None:
        rng = np.random.default_rng(shift_seed(random_seed, 1234567))

    if n_classes == 1:  # will I continue with this 1-classes distinction??
        n_classes = 2
    if n_classes is None:
        n_classes = len(np.unique(X))
    if n_classes < 2:
        # If we don't have at least 2 distinct classes, there is no information
        return np.zeros(X.shape[0])

    assert X.shape[-1] >= np.square(
        n_classes
    ), f"Must have ensemble of size at least {np.square(n_classes)}, whereas yours is size {X.shape[-1]}."

    batch_size = X.shape[-2]
    # for now, aways take sub-batches of size so that in principle
    # the ensemble could find all the information
    sub_batch_size = n_classes ** int(np.log(batch_size) / np.log(n_classes))

    # create subsampling indices
    sub_indices = np.zeros(batch_size * (batch_size - 1) / 2)


def group_sizes(size, group_size):
    sizes = int(size / group_size) * [group_size]
    if size % group_size > 0:
        stub = size % group_size
        for i in cycle(range(len(sizes))):
            if sizes[i] <= stub:
                break
            sizes[i] -= 1
            stub += 1
        sizes.append(stub)
    return sizes


def group_edges(size, group_size):
    sizes = group_sizes(size, group_size)
    edges = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    return list(zip(edges[:-1], edges[1:]))
