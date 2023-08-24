import sys

import numpy as np
import scipy.linalg
from scipy.optimize import newton

from ..decorators import flatten_batch, get_defaults_from_self
from ..stats import covariance_from_similarity, similarity_exp
from ..utils import concatenate
from .selector import UncertaintySelector, optimize_batch


# pylint: disable=invalid-name
@flatten_batch(bdim=-2, to_flatten=2)
def solve_triangular_torch(A, b, lower=False, trans=0, **kwargs):
    """Solve a linear system using Pytorch."""
    import torch

    A = torch.transpose(torch.tensor(A), -2, -1) if trans == 1 or trans == "T" else torch.tensor(A)
    return torch.linalg.solve_triangular(
        A, b if isinstance(b, torch.Tensor) else torch.tensor(b), upper=not lower, **kwargs
    )


# pylint: disable=invalid-name
@flatten_batch(bdim=-2, to_flatten=2)
def solve_triangular_scipy(A, b, **kwargs):
    """Solve a linear system using SciPy."""
    soln = np.empty(b.shape)

    for i in range(A.shape[0]):
        soln[i] = scipy.linalg.solve_triangular(A[i], b[i], check_finite=False, **kwargs)

    return soln


HAS_TORCH = None


def solve_triangular(*args, **kwargs):
    global HAS_TORCH
    if HAS_TORCH is None:
        HAS_TORCH = "torch" in sys.modules
    return (solve_triangular_torch if HAS_TORCH else solve_triangular_scipy)(
        *args, **kwargs
    )


class CovarianceSelector(UncertaintySelector):
    """
    Batch selector which looks for batches with large total covariance, i.e., large
    joint entropy.

    :param model: An instance of :class:`models.CovarianceRegressor`. Will be used to
        determine prediction covariances for proposed batches.
    :param samples: The sample pool to select from. Can be a numpy-style addressable
        array (with first dimension indexing samples, and other dimensions indexing
        features)---note that :class:`data.Dataset` serves this purpose---or an instance of
        :class:`sample_generation.SampleGenerator`, in which case the `num_samples` parameter is
        in effect.
    :param num_samples: If a :class:`SampleGenerator` has been provided via the `samples`
        parameter, then at the start of a call to :meth:`.select`, `num_samples`
        samples will be drawn from the `SampleGenerator`, or as many samples as the
        :class:`SampleGenerator` can provide, whichever is less. Defaults to :Inf:, i.e., draws
        as many samples as available.
    :param batch_size: Size of the batch to select.
    :param regularization: The diagonal of the covariance matrix will be multiplied by
        (1 + regularization), after being computed by the model. This ensures that
        the coviarance matrix is positive definite (as long as all the covariances
        are positive). Defaults to .05

        This parameter is particularly important if the covariance is computed from
        an ensemble of models, and the ensemble is not very large: for a given batch
        of N samples, and a model ensemble size of M, the distribution of predictions
        will consist of M points in an N-dimensional space. If M < N, the covariance
        of the batch predictions is sure to have determinant 0, and no comparisons
        can be made (without regularization). Even if M >= N, a relatively small
        ensemble size can produce numerical instability in the covariances, which
        regularization smooths out.
    :param normalize: If True, scales the (co)variances by the inverse-square-length
        of the embedding vector (retrieved by a call to `model.embedding`), modulo a
        small constant. This prevents the algorithm from just seeking out those inputs
        which give large embedding vectors. Defaults to False.

        If the model has not implemented an `.embedding` method, setting
        `normalize = True` will raise a `NotImplementedError` when you call
        :meth:`.select`. :class:`LaplaceApproxRegressor` and subclasses have implemented
        :meth:`.embedding`, as have some others.
    :param normalize_epsilon: In the normalization step described above, variances are
        divided by |embedding_length|² + ε, where
            ε = normalize_epsilon * MEAN_SQUARE(all embedding lengths)
        Defaults to 1e-3. Should be related to measurement error.
    :param similarity: The effective covariance matrix (before regularization) will be
        (1 - similarity) * covariance (computed from the model) + similarity * S,
        where S is a "synthetic" covariance computed from a similarity matrix, as
        follows:
        First, a similarity matrix is computed: each feature dimension in the data, X,
        will be normalized to have variance of 1. Then, a euclidean distance matrix is
        computed for the whole dataset (divided by sqrt(N), where N is the number of
        feature dimensions). Then, this distance metric is passed into a decaying
        exponential, with 1/e-life equal to the parameter 'similarity_scale'. This
        gives a positive-definite similarity matrix, with ones on the diagonal.
        Second, the similarity matrix is interpreted as a correlation matrix.
        Variances are taken from the model (i.e., copied from the diagonal of the
        covariance matrix). Together, these data determine a covariance matrix, which
        will be combined with the model covariance in the given proportion.
        Defaults to 0.
    :param similarity_scale: This tunes the correlation matrix in the similarity
        computation above. The pairwise euclidean distances in normalized feature
        space are passed into a exponential with 1/e-life equal to similarity_scale.
        So, a smaller value for similarity_scale will give smaller off-diagonal
        entries on the correlation/covariance matrix.
        If similarity_scale is set to 'auto', then a scale is chosen to match the
        mean squares of the similarities and the correlations (from the model).
    :param prior: Specifies a "prior probability" for each sample. The covariance
        matrix (after the application of similarity) will be multiplied by the priors
        such that, if C_ij is a covariance and p_i, p_j are the corresponding priors,
        C'_ij = C_ij p_i p_j. This is a covenient way of introducing factors
        other than covariance into the ranking. 'prior' may be an array of numbers
        (of size num_samples), or a function (applied to the samples), or one of the
        following:
            'prediction': calculates a prior from the quantile of the predicted
                performance (not the uncertainties). 'prior_scale' sets the power
                this quantile is raised to.
        Defaults to the constant value 1.
    :param prior_scale: The prior will be raised to this power before applying it to
        the covariance. Defaults to 1.
    :param prefilter: Reduces the incoming sample pool before applying batch
        selection. Selects the subset of the samples which have the highest
        std_dev * prior score. If 0 < prefilter < 1, takes this fraction of the
        sample pool. If prefilter >= 1, takes this many samples. Since batch
        selection computes and stores the size-N^2 covariance matrix for the whole
        sample pool, it should work with at most around 10,000 samples. Prefiltering
        can work with much larger pools, since it only needs to compute N standard
        deviations. Therefore, a practical strategy is to take a sample pool about
        5 times as big as you can handle covariances for, then narrow down to only
        the top 20% individual scores before batch selection. Narrowing to much less
        than 20% risks changing what will ultimately be the optimal batch.
    :param random_seed: A random seed for deterministic behaviour.
    """

    def __init__(
        self,
        model=None,
        samples=None,
        num_samples=float("inf"),
        batch_size=1,
        normalize=False,
        normalize_epsilon=1e-3,
        regularization=0.05,
        similarity=0,
        similarity_scale="auto",
        prior=1,
        prior_scale=1,
        prefilter=None,
        random_seed=None,
        fast_opt=True,
        n_rounds=10,
        **kwargs
    ):
        super().__init__(
            model=model,
            # labelled_samples=labelled_samples,
            samples=samples,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        self.normalize = normalize
        self.normalize_epsilon = normalize_epsilon
        self.regularization = regularization
        self.similarity = similarity
        self.similarity_scale = similarity_scale
        self.rng = np.random.default_rng(random_seed)
        self.prior = prior
        self.prior_scale = prior_scale
        self.prefilter = prefilter
        self.n_rounds = n_rounds
        self.fast_opt = fast_opt

    @get_defaults_from_self
    def _select(
        self,
        samples=None,
        fixed_samples=None,
        batch_size=None,
        prior=None,
        fixed_prior=None,
        verbose=False,
        **kwargs
    ):
        if fixed_samples is not None:
            n_fixed = len(fixed_samples)
            samples = concatenate(fixed_samples, samples)
        else:
            n_fixed = 0

        # get prediction covariances
        if verbose:
            print("Computing covariance matrix for sample pool...")
        cov = self.model.covariance(samples)

        # smooth out the covariance with a similarity term
        if self.similarity:
            if verbose:
                print("Computing similarity matrix...")
            if self.similarity_scale == "auto":
                # similarity is scaled so that it has the same mean square
                # as the correlation
                # only subsamples
                sim_1 = similarity_exp(samples, scale=1)
                std_dev = np.sqrt(cov.diagonal())
                corr = cov / std_dev[None, :] / std_dev[:, None]
                corr_mean = np.mean(np.square(corr))
                log_s1 = np.log(sim_1)

                def f(x):
                    return np.mean(np.power(sim_1, 2 * x)) - corr_mean

                def df(x):
                    return 2 * np.mean(np.power(sim_1, 2 * x) * log_s1)

                x = newton(f, 1.0, df, tol=corr_mean * 1e-4, full_output=True, disp=False)[0]
                sim = np.power(sim_1, x)
            else:
                sim = similarity_exp(samples, scale=self.similarity_scale)

            cov = (1 - self.similarity) * cov + self.similarity * covariance_from_similarity(
                sim, cov.diagonal()
            )

        # normalize covariance for embedding vector size
        if self.normalize and hasattr(self.model, "embedding"):
            if verbose:
                print("Normalizing covariances by embedding norms...")
            emb_norm2 = (self.model.embedding(samples) ** 2).sum(axis=-1)
            eps = self.normalize_epsilon * emb_norm2.mean()
            prior = prior / np.sqrt(emb_norm2 + eps)

        # apply the prior
        cov = prior[None, :] * cov * prior[:, None]

        epsilon = np.mean(np.diag(cov)) * 1e-8

        # This actually returns 1/2 the log-determinant of the covariance
        def cov_fn(indices):
            # select sub-matrix of covariance matrix
            i_0, i_1 = np.broadcast_arrays(indices[..., None, :], indices[..., :, None])
            cov_batch = cov[i_0, i_1]
            assert cov_batch.shape == (*indices.shape, indices.shape[-1])

            # regularize to avoid non-positive matrices
            # (eg., when a sample is repeated in a batch)
            batch_indices = np.arange(indices.shape[-1])
            cov_batch[..., batch_indices, batch_indices] += (
                cov_batch[..., batch_indices, batch_indices] * self.regularization + epsilon
            )

            # compute log-determinant using cholesky decomposition
            R = np.linalg.cholesky(cov_batch)
            return np.sum(np.log(np.diagonal(R, axis1=-2, axis2=-1)), axis=-1)

        # This is a weaker version...
        def cov_fn_opt_step(indices, samples):
            # select sub-matrix of covariance matrix
            i_0, i_1 = np.broadcast_arrays(indices[..., None, :], indices[..., :, None])
            cov_batch = cov[i_0, i_1]
            assert cov_batch.shape == (*indices.shape, indices.shape[-1])

            # regularize to avoid non-positive matrices
            # (eg., when a sample is repeated in a batch)
            batch_indices = np.arange(indices.shape[-1])
            cov_batch[..., batch_indices, batch_indices] *= 1 + self.regularization

            # pick out covariances of each batch w.r.t. the whole sample_space
            C10 = cov[indices, :]
            assert C10.shape == (*indices.shape, len(samples))

            L00 = np.linalg.cholesky(cov_batch)
            L10 = np.asarray(solve_triangular(L00, C10, lower=True))
            assert L10.shape == (*indices.shape, len(samples))
            return np.swapaxes(np.diag(cov) - np.sum(L10**2, axis=1), 0, 1)

        if verbose:
            print("Optimizing batches...")
        batch_indices = optimize_batch(
            cov_fn,
            batch_size,
            len(samples),
            n_fixed=n_fixed,
            scoring_opt_step=cov_fn_opt_step if self.fast_opt else None,
            n_rounds=self.n_rounds,
            random_seed=self.rng.integers(1e8),
            # n_tuples=None, n_best=None,
            verbose=verbose,
        )

        return batch_indices
