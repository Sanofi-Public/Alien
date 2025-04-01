"""
Have some intro lines here.
"""

import gc
import sys
from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import cycle

import numpy

from .decorators import flatten_batch, normalize_args
from .tumpy import tumpy as np
from .utils import Peekable, diagonal, ranges

# pylint: disable=no-value-for-parameter


class Output(Enum):
    CLASS = "class"
    PROB = "prob"
    LOGIT = "logit"


def get_output_type(value):
    if isinstance(value, str):
        return {"class": Output.CLASS, "prob": Output.PROB, "logit": Output.LOGIT}[value.lower()[:5]]
    return Output(value)


@flatten_batch(bdim=-1)
def multiply_cov(ensemble, multiple: float):
    """
    Modifies an ensemble to change its covariance by
    a factor of 'multiple'.

    :param ensemble: ensembles are over the last dimension
        only.
    """
    if multiple == 1:
        return ensemble
    means = np.mean(ensemble, axis=-1, keepdims=True)
    if isinstance(ensemble, numpy.ndarray):
        means = numpy.asarray(means)
    ensemble = multiple * ensemble + (1 - multiple) * means
    return ensemble


@flatten_batch(bdim=-1)
def augment_ensemble(preds, N: int, multiple=1.0, rng=None, epsilon=1e-6):
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
        preds = multiply_cov(preds, multiple)

    if N <= preds.shape[-1]:
        return preds

    mean = np.mean(preds, axis=-1)
    cov = covariance_from_ensemble(preds)
    aug = ensemble_from_covariance(mean, cov, N - preds.shape[-1], rng, epsilon=epsilon)
    return np.concatenate((preds, aug), axis=-1)


def rankdata(x):
    """
    Returns the ranking of each element in `x`, with `1` being the smallest.
    Ties are broken nondeterministically.
    """
    return np.unique(x, return_inverse=True)[1] + 1


# ---- Computing (co)variances from ensembles ---- #


def covariance_from_ensemble(  # NOSONAR
    p0, p1=None, block_size=None, indices=None, generate=None, weights=None, ddof=1
):  # NOSONAR
    """
    Computes the covariance of an ensemble, or between two ensembles.
    First axis of `p0` is batch, second axis is ensemble, higher axes are outputs.

    Args:
        p0 (ndarray, Tensor): The first (and possibly only) ensemble.
            If only `p0` is provided, computes the covariance of `p0` with itself.
            Shape should be `(n_variables_0, ensemble_size, *out_shape_0)`.
        p1 (ndarray, Tensor): The second ensemble. If nonzero, computes the covariance
            between `p0` and `p1`. Shape should be `(n_variables_1, ensemble_size, *out_shape_1)`.
        block_size (int): If not `None`, computes the covariance in blocks of size `(block_size, block_size)`.
        indices (iterable): If not `None`, an iterable of pairs of indices giving the order in
            which the covariances are computed (and the order in which they are yielded,
            if `generate` is True).
        generate (bool): if True, returns an iterator through the lower triangle of the covariance
            matrix (or through `p0[indices]` if `indices ≠ None`).
            The first item yielded is a list of diagonal covariances, i.e., variances.
        ddof (int): The delta degrees of freedom for the covariance matrix. Default is 1.

    Returns:
        Array/Tensor of size `(n_variables_0, n_variables_1, *out_shape_0, *out_shape_1)`
    """
    if weights is not None:
        raise NotImplementedError
    device = np.device(p0)
    p0 = np.to(p0, device)
    p1 = np.to(p1, device) if p1 is not None else p1

    if p0.ndim == 2 and not block_size:
        if p1 is None:
            l = len(p0)
            return np.cov(p0, ddof=ddof).reshape(l, l)
        else:
            p0, p1 = p0 - p0.mean(axis=1)[:, None], p1 - p1.mean(axis=1)[:, None]
            return p0 @ p1.T / (p0.shape[1] - ddof)

    diag = p1 is None
    p1 = p0 if diag else p1
    N_0, N_1 = len(p0), len(p1)

    if not block_size:
        from . import default_compute_block_size

        block_size = default_compute_block_size

    if block_size:  # preds.ndim >= 2
        assert p1 is p0, "block computation doesn't yet work between two sets of variables"
        preds = p0

        blocks0 = ranges(N_0, block_size)
        blocks1 = blocks0 if diag else ranges(N_1, block_size)
        if indices is None:
            indices = zip(*np.tril_indices(len(blocks0), -1)) if diag else np.ndindex(len(blocks0), len(blocks1))

        def generator_cfe_1():
            if diag:
                # print(np.var(preds[0:1000], axis=1).mean().sqrt())
                yield [covariance_from_ensemble(preds[i0:i1], ddof=ddof) for i0, i1 in blocks0]

            for i, j in indices:
                yield covariance_from_ensemble(preds[slice(*blocks0[i])], preds[slice(*blocks1[j])], ddof=ddof)

        return generator_cfe_1()

    # Not working in blocks:

    if isinstance(p0, list) or p0.dtype == object:

        if indices is None:
            indices = zip(*np.tril_indices(N_0, -1)) if diag else np.ndindex(N_0, N_1)

        if generate:

            def generator_cfe_2():
                if diag:
                    yield [covariance_from_ensemble(p[None, ...], ddof=ddof) for p in p0]

                for i, j in indices:
                    yield covariance_from_ensemble(p0[i][None, ...], p1[j][None, ...], ddof=ddof)

            return generator_cfe_2()

        else:
            cov = np.empty((N_0, N_1), dtype=object, device=device)
            if diag:
                for i in range(N_0):
                    cov[i, i] = covariance_from_ensemble(p0[i], ddof=ddof)
            for i, j in indices:
                cov[i, j] = covariance_from_ensemble(p1[j], p0[i], ddof=ddof)
                if diag:
                    cov[j, i] = cov[i, j].T
            return cov

    else:
        shape0, shape1 = p0.shape[2:], p1.shape[2:]
        p0, p1 = np.moveaxis(p0, 1, -1), np.moveaxis(p1, 1, -1)  # ensemble is now last dimension
        p0, p1 = p0.reshape(-1, p0.shape[-1]), p1.reshape(-1, p1.shape[-1])
        cov = covariance_from_ensemble(p0, p1, ddof=ddof)
        cov = np.moveaxis(cov.reshape(N_0, *shape0, N_1, *shape1), len(shape0) + 1, 1)
        return cov


def variance_from_ensemble(preds, pbar=False, iterator=False):
    """Variance of the ensemble `preds`

    If output is of dimension `k`,
    returns the `k x k` internal covariance of each prediction.

    Args:
        preds (ndarray, Tensor, list): The ensemble of predictions.
            If preds is an `ndarray` or `Tensor`, the first axis is the batch dimension,
            the second axis is the ensemble dimension, and the remaining axes are the features.
            If preds is a list, each element is an array-like, now with the first axis being the
            ensemble dimension.
        pbar (bool): If True, shows a progress bar.
        iterator (bool): If True, returns an iterator over the variances.

    """
    if isinstance(preds, (list, Iterator)) or preds.dtype == object:
        if pbar:
            from tqdm import tqdm

            preds = tqdm(preds, desc="Variances")

        def generator_vfe():
            for _, p in enumerate(preds):
                v = variance_from_ensemble(p[None, ...])
                yield v

        return generator_vfe() if iterator else np.asarray(list(generator_vfe()))

    if preds.ndim == 2:
        return np.var(preds, axis=1)

    shape = preds.shape
    preds = preds.reshape(shape[0], shape[1], -1)
    preds = preds - preds.mean(axis=1, keepdims=True)
    return (preds[..., None, :] * preds[..., None]).mean(axis=1)


def std_dev_from_ensemble(  # NOSONAR
    preds, log=False, rel_epsilon=1e-4, epsilon=None, weights=None, mean=True, verbose=False, device=None
):
    """
    If `preds` is multivariate, returns the determinant of the internal
    covariance of each `preds[i]`, regularized by adding `epsilon` to the
    diagonal. If `epsilon` is `None`, it is computed as::

        rel_epsilon = epsilon * mean(diagonal)

    If `log` is True, this returns the logarithm (base-2) of the std-dev/determinant. This is
    more numerically stable than computing the determinant first and then the log, since it
    uses the Cholesky decomposition to compute log-det directly.
    """
    assert weights is None, "Not implemented yet!"

    if isinstance(preds, (list, Iterable, Iterator)) or preds.dtype == object:
        std = np.zeros(len(preds), dtype=np.float32, device=device)
        if verbose:
            print("Computing variances...")
        var = Peekable(variance_from_ensemble(preds, pbar=False, iterator=True))
        if epsilon is None:
            # reg = 0.
            # for v in var:
            #    reg += v.trace() / len(v)
            # epsilon = rel_epsilon * reg / len(var)
            epsilon = rel_epsilon * var.peek().trace() / len(var.peek())

        if verbose:
            print("Computing standard devs...")
            from tqdm import tqdm

            var = tqdm(var, desc="Std Dev", total=len(preds))

        clear_torch = np.backend == "torch"
        if clear_torch:
            import torch

        for i, v in enumerate(var):
            if epsilon > 0.0:
                diagonal(v)[:] += epsilon
            log_chol = 0.5 * np.log2(np.linalg.cholesky(v).diagonal(axis1=-2, axis2=-1))
            std[i] = log_chol.mean(axis=-1) if mean else log_chol.sum(axis=-1)

            if clear_torch:
                del v
                del log_chol
                gc.collect()
                torch.cuda.empty_cache()

        return std if log else np.exp2(std)

    if preds.ndim == 2:
        return np.std(preds, axis=1) if not log else np.log2(np.std(preds, axis=1))

    shape = preds.shape
    var = variance_from_ensemble(preds.reshape(shape[0], shape[1], -1))

    if epsilon is None:
        epsilon = rel_epsilon * np.mean(np.trace(var, axis1=-2, axis2=-1)) / shape[-1]
    if epsilon > 0.0:
        diagonal(var)[:] += epsilon  # diagonal over last two axes

    entropy_ = 0.5 * np.log2(
        np.linalg.cholesky(var).diagonal(axis1=-2, axis2=-1).sum(axis=-1)
    )  # .reshape(shape[0], *shape[2:])

    return entropy_ if log else np.exp2(entropy_)


# ---- Computing entropies from (co)variance ---- #


def entropy_from_covariance(cov, epsilon=None, rel_epsilon=1e-4):
    """
    This is batch-compatible.
    """
    if epsilon is None:
        epsilon = rel_epsilon * cov.trace().mean() / cov.shape[-1]
    if epsilon > 0.0:
        cov = cov.copy()
        diagonal(cov)[:] += epsilon

    return 0.5 * np.log2(np.linalg.cholesky(cov).diagonal(axis1=-2, axis2=-1)).sum(axis=-1)


def joint_entropy_2x2_from_covariance(v_0, v_1, c, epsilon=None, rel_epsilon=1e-4):
    """
    `v_0` and `v_1` are the variances (as square matrices) of the two samples,
    and `c` is the covariance between them.

    If `v_0` is MxM and `v_1` is NxN, then `c` should be NxM. I.e., `v_0` and `c`
    should concatenate along axis 0.

    `epsilon` regularizes the covariance, by adding epsilon * Id.

    Returns:
        The joint entropy, a single floating point number
    """
    cov = np.concatenate((np.concatenate((v_0, c), axis=0), np.concatenate((c.T, v_1), axis=0)), axis=1)

    if epsilon is None:
        epsilon = rel_epsilon * cov.trace() / len(cov)
    if epsilon > 0.0:
        diagonal(cov)[:] += epsilon

    return entropy_from_covariance(cov, epsilon=epsilon)


def joint_entropy_from_covariance(  # NOSONAR
    cov,
    diag=False,
    epsilon=None,
    rel_epsilon=1e-4,
    generate=False,
    indices=None,
    triangle=True,
    pbar=False,
    device=None,
):
    """
    Computes a joint entropy matrix from a covariance matrix.
    If `cov.dtype` is `object`, or `cov.ndim` > 2, it will take the first
    two dimensions of `cov` as indexing the samples, and the remaining
    dimensions as indexing the output dimensions.

    `epsilon` regularizes the covariance, by adding epsilon * Id. If
    epsilon is None, `epsilon = rel_epsilon * mean(diagonal) * Id.

    If `triangle` is True, fills in only the lower triangle of the matrix.
    Otherwise, fills it in symmetrically.

    Args:
        cov (np.ndarray, torch.Tensor, Iterator): The covariance matrix.
            The first two dimensions of `cov` index the different variables to be paired,
            and the remaining dimensions index the output dimensions, in consecutive pairs.

            If `cov` is an iterator, then the first item yielded should itself be a list of the diagonal
            blocks, and then the remaining items are the lower-triangular blocks.
            Note that the function `covariance_from_ensemble` returns an iterator
            which can be passed directly to this function.
        diag (bool): If True, computes the diagonal elements, i.e., the entropy of each
            variable individually.
        epsilon (float): Regularizes the covariance, by adding `epsilon * Id`. Default
            is `None`.
        rel_epsilon (float): If `epsilon` is `None` (the default), then `epsilon = rel_epsilon * mean(diagonal) * Id`.
            `rel_epsilon` is ignored if `epsilon` is not `None`. Default is `1e-4`.
        generate (bool): If True, returns an iterator through the lower triangle of the joint entropy matrix. Default is `False`.
        indices (iterable): If not `None`, an iterable of pairs of indices giving the order in
            which the joint entropies are computed, and the order in which they are yielded,
            if `generate` is True.
        triangle (bool): If True, fills in only the lower triangle of the matrix. Otherwise, fills it in symmetrically.
        pbar (bool): If True, shows a progress bar.
        device: If using Pytorch, the device to use. If using Numpy, this is ignored.

    Returns:
        If `generate` is False (the default), returns the joint entropy matrix.
        If `generate` is True, returns an iterator over the lower triangle of the joint
        entropy matrix, in the order given by `indices` (or the default lower-triangular order).
    """
    assert triangle, (
        "Filling in the full symmetric matrix is not implemented yet."
        "Set `triangle=True` to fill only the lower left triangle."
    )

    # `cov` is an iterator which yields blocks of the covariance matrix
    if isinstance(cov, Iterator):
        diag = next(cov)
        n_blocks = len(diag)
        if device is None:
            device = np.device(diag[0])

        block_sizes = np.fromiter(map(len, diag), dtype=np.int32, count=n_blocks, device=device)
        block_edges = np.zeros((len(block_sizes), 2), dtype=np.int32, device=device)
        block_edges[:, 1] = np.cumsum(block_sizes)
        block_edges[1:, 0] = block_edges[:-1, 1]
        N = block_edges[-1, 1]

        outlen = int((diag[0].ndim - 2) / 2)
        outshape = diag[0].shape[2 : 2 + outlen]
        outsize = 1 if outlen == 0 else np.prod(outshape)

        bs = len(diag[0])
        diag = [np.to(c.reshape(len(c), len(c), outsize, outsize), device) for c in diag]

        if epsilon is None:
            if rel_epsilon:
                reg = sum(c.trace().trace() / (len(c) * outsize) for c in diag)
                epsilon = rel_epsilon * reg / len(block_sizes)
            else:
                epsilon = 0.0

        if epsilon:
            for c in diag:
                diagonal(diagonal(c, axes=(2, 3)), axes=(0, 1))[:] += epsilon

        jen = np.zeros((N, N), dtype=diag[0].dtype, device=device)

        if pbar:
            from tqdm import tqdm

            pbar = tqdm(total=int(n_blocks * (n_blocks + 1) / 2))

        for c_ij, i, j in zip(cov, *np.tril_indices(n_blocks, -1)):
            c_ij = np.to(c_ij, device)
            n_i, n_j = c_ij.shape[:2]  ###############
            c_ij = c_ij.reshape(n_i, n_j, outsize, outsize)  ##########

            d_ii = np.moveaxis(diag[i].diagonal(), -1, 0)  # ni, outsize, outsize
            d_ii = np.broadcast_to(d_ii[:, None, ...], (n_i, n_j, outsize, outsize))
            d_jj = np.moveaxis(diag[j].diagonal(), -1, 0)  # nj, outsize, outsize
            d_jj = np.broadcast_to(d_jj[None, ...], (n_i, n_j, outsize, outsize))

            # a doubling of cov
            cov_2x2 = np.zeros((n_i, n_j, 2 * outsize, 2 * outsize), dtype=c.dtype, device=device)

            # for the lower triangle, i > j, so
            # jj_diag should be in the upper left
            cov_2x2[:, :, outsize:, outsize:] = d_ii
            cov_2x2[:, :, :outsize, :outsize] = d_jj

            # only fill in the lower-triangle,
            # where first index is bigger than second:
            cov_2x2[:, :, outsize:, :outsize] = c_ij  ########### .swapaxes(0,1).swapaxes(2,3)

            # maybe we have to fill in the upper triangle?
            # cov_2x2[:,:,:outsize,outsize:] = c_ij

            jen[block_edges[i, 0] : block_edges[i, 1], block_edges[j, 0] : block_edges[j, 1]] = entropy_from_covariance(
                cov_2x2
            )

            if pbar:
                pbar.update(1)

        # must fill in diagonal blocks
        for block, (i_0, i_1) in zip(diag, block_edges):
            jen[i_0:i_1, i_0:i_1] = joint_entropy_from_covariance(block, epsilon=0, triangle=triangle)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return jen

    if device is None:
        device = np.device(cov)
    else:
        cov = np.to(cov, device)

    if cov.dtype != object:
        # right now, cov.shape = (N, N, *outshape, *outshape)
        # find out how big *outshape is:
        N = len(cov)
        outlen = int((cov.ndim - 2) / 2)
        outsize = 1 if outlen == 0 else np.prod(cov.shape[2 : 2 + outlen])

        cov = np.to(cov.reshape(N, N, outsize, outsize), device=device)
        cov_diag = diagonal(diagonal(cov, axes=(2, 3)), axes=(0, 1))

        if epsilon is None:
            epsilon = rel_epsilon * cov_diag.mean()
        if epsilon > 0:
            cov_diag[:] += epsilon

        diag = np.moveaxis(cov.diagonal(), -1, 0)
        ii_diag = np.broadcast_to(diag[None, ...], (N, N, outsize, outsize))
        jj_diag = np.swapaxes(ii_diag, 0, 1)

        # a 2x2 blow-up of cov
        block_cov = np.zeros((N, N, 2 * outsize, 2 * outsize), dtype=diag.dtype, device=device)

        # for the lower triangle, i > j, so
        # jj_diag should be in the upper left
        block_cov[:, :, outsize:, outsize:] = ii_diag
        block_cov[:, :, :outsize, :outsize] = jj_diag

        # only fill in the lower-triangle,
        # where first index is bigger than second:
        block_cov[:, :, outsize:, :outsize] = cov.swapaxes(2, 3)

        # zeroing ij_cov and ji_cov when i==j
        # We don't actually care about the entropies on the diagonal,
        # we just don't want a non-positive-definite matrix
        block_diag = diagonal(block_cov, axes=(0, 1))
        block_diag[:outsize, outsize:] = 0
        block_diag[outsize:, :outsize] = 0

        return entropy_from_covariance(block_cov)

    else:  # cov.dtype == object
        # preload the diagonal, which gets used often:
        diag_cov = cov.diagonal()
        N = len(diag_cov)

        if epsilon is None:
            reg = 0.0
            for c in diag_cov:
                reg += c.trace() / len(c)
            epsilon = rel_epsilon * reg / N

        if not generate:
            jen = np.zeros((N, N), dtype=diag_cov[0].dtype, device=device)

        if diag or epsilon:
            for i in range(N):
                if epsilon:
                    diagonal(diag_cov[i])[:] += epsilon
                if diag and not generate:
                    jen[i, i] = entropy_from_covariance(diag_cov[i], epsilon=0)

        if indices is None:
            indices = zip(*np.tril_indices(N, -1))

        if not isinstance(cov, Iterator):
            cov_iter = (cov[i, j] for i, j in indices)
        else:
            cov_iter = cov

        if generate:

            def generator_jefc():
                for (i, j), c in zip(indices, cov_iter):
                    yield joint_entropy_2x2_from_covariance(diag_cov[i], diag_cov[j], c, epsilon=0)
                return

            return generator_jefc()

        for (i, j), c in zip(indices, cov_iter):
            jen[i, j] = joint_entropy_2x2_from_covariance(diag_cov[i], diag_cov[j], c, epsilon=0)
            if not triangle:
                jen[j, i] = jen[i, j]

        return jen


# ---- Computing entropies from regression ensembles ---- #


def joint_entropy_from_ensemble(
    preds,
    epsilon=None,
    rel_epsilon=1e-4,
    generate=False,
    reduce_pc=None,
    stack_pc=True,
    ddof=1,
    block_size=None,
    pbar=False,
):
    """
    Computes the pairwise joint entropies of the regression ensemble `preds`, returning them
    as a square matrix.

    Args:
        preds (np.ndarray, torch.Tensor, list): The ensemble of predictions.
        reduce_pc (bool, int): If a positive integer, does a PCA reduction to the high-dimensional
            outputs
        stack_pc (bool): If True, and if preds is a list or has dtype=object, stacks the
            results of the PCA before computing joint entropies, which should be possible
            now, since they're all of the same size.
        epsilon (float): Regularizes the covariance, by adding `epsilon * Id`. Default
            is `None`.
        rel_epsilon (float): If `epsilon` is `None` (the default), then `epsilon = rel_epsilon * mean(diagonal) * Id`.
            `rel_epsilon` is ignored if `epsilon` is not `None`. Default is `1e-4`.
        generate (bool): If True, returns an iterator through the lower triangle of the joint entropy matrix.
        ddof (int): The delta degrees of freedom for the covariance matrix. Default is 1.
        block_size (int): If not `None`, computes the joint entropy in blocks of size `(block_size, block_size)`.
        pbar (bool): If True, shows a progress bar.
    """
    if reduce_pc:
        preds = reduce_ensemble(preds, reduce_pc, stack=stack_pc)

    return joint_entropy_from_covariance(
        covariance_from_ensemble(preds, generate=generate, ddof=ddof, block_size=block_size),
        epsilon=epsilon,
        rel_epsilon=rel_epsilon,
        pbar=pbar,
    )


def entropy_from_ensemble(preds, epsilon=None, rel_epsilon=1e-4):
    """Returns the entropy of the whole batch"""
    return entropy_from_covariance(covariance_from_ensemble(preds), epsilon=epsilon, rel_epsilon=rel_epsilon)


# -------------------------------- #


def reduce_ensemble(preds, n_components, top_variance=False, stack=True):
    """
    Applies PCA reduction to the ensemble `preds`. Accepts exactly 1 batch dimension.

    Args:
        preds (np.ndarray, torch.Tensor, list): The ensemble of predictions.
            If preds is an `ndarray` or `Tensor`, the first axis is the batch dimension,
            the second axis is the ensemble dimension, and the remaining axes are the features.
            If preds is a list, each element is an array-like, now with the first axis being the
            ensemble dimension.
        n_components (int): The number of components to reduce the ensemble to.
        top_variance (bool): If True, first does a quick and dirty cull, taking only
            `top_variance' features based on their variance, before reducing the remaining
            variables with PCA. This can speed up a slow PCA computation, at the cost of accuracy.
    """
    if isinstance(preds, Iterator) or isinstance(preds, list) or getattr(preds, "dtype", None) == object:
        reds = [apply_pca(p, n_components, sample_axis=0, top_variance=top_variance) for p in preds]
        return np.stack(reds) if stack else reds
    return apply_pca(preds, n_components, sample_axis=1, top_variance=top_variance)


def apply_pca(x, n_components, sample_axis=0, top_variance=False):
    """
    Applies PCA reduction to the ensemble `x` represented as an array or tensor.

    Args:

        x (np.ndarray, torch.Tensor): An ensemble of predictions. `sample_axis` indexes the samples,
            later axes index the features, earlier axes index within the batch (so by default,
            there is no batch dimension).
        n_components (int): The number of components to reduce the ensemble to.
        sample_axis (int): The axis of `x` that indexes within the ensemble.
        top_variance (int, float): If this is an integer, first does a quick and dirty cull, taking only
            `top_variance' features based on their variance, before reducing the remaining
            variables with PCA. This can speed up a slow PCA computation, at the cost of accuracy.
    """
    if x.ndim - sample_axis > 2:
        x = x.reshape(*x.shape[: sample_axis + 1], -1)

    if top_variance:
        # variance = np.var(x, axis=-2)
        # top_i = topk(variance, top_variance, axis=-1)
        # x = x[..., top_i]
        raise NotImplementedError("No variance-culling until we've tested basic PCA")

    if "torch" in sys.modules:
        import torch

        # https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        U, S, V = torch.pca_lowrank(torch.as_tensor(x), q=min(n_components, x.shape[1]))
        out = torch.matmul(torch.as_tensor(x), V)
        if out.shape[-1] < n_components:
            print(f"Very few ({x.shape[1]}) features, less than {n_components = }")
            out = torch.cat((out, torch.zeros((out.shape[0], n_components - out.shape[1]), dtype=out.dtype)), dim=1)

    else:
        from sklearn.decomposition import PCA

        batch_shape = x.shape[:sample_axis]
        x = x.reshape((-1, *x.shape[sample_axis:]))
        out = np.empty((*x.shape[:2], n_components), dtype=x.dtype)
        for i in range(x.shape[0]):
            out[i] = PCA(n_components=min(n_components, x.shape[1]), svd_solver="randomized").fit_transform(x[i])
        if out.shape[-1] < n_components:
            print(f"Very few ({x.shape[-1]}) features, less than {n_components = }")
            out = np.concatenate((out, np.zeros((out.shape[0], n_components - out.shape[1]), dtype=out.dtype)), axis=1)
        out = out.reshape((*batch_shape, *out.shape[1:]))

    return out


def topk(x, k, axis=None, largest=True):
    if "torch" in sys.modules:
        import torch

        return torch.topk(x, k, dim=axis, sorted=False, largest=largest).indices
    else:
        import numpy

        if largest:
            k = -k
        return numpy.argpartition(x, k)[:k]


# @flatten_batch(bdim=-2, to_flatten=2, degree)
def ensemble_from_covariance(mean, cov, ensemble_size, rng=None, random_seed=None, epsilon=0.):
    """ """
    #breakpoint()
    if rng is None:
        rng = np.random.default_rng(random_seed)
    if epsilon:
        cov = cov.copy()
        diag = diagonal(cov)
        diag[:] += epsilon * diag.mean()

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

    #   Restricts X to only look at 'relevant_features'.

    #   If 'normalize' is True, rescales the input to have standard
        deviation 1 in each feature, then rescales by another factor
        of sqrt(X.shape[-1]).

    #   Then the distance matrix is put into a Gaussian of std dev
        equal to 'scale'.

    """
    if relevant_features is not None:
        X = X[..., relevant_features]

    distance = np.linalg.norm(X[..., :, None, :] - X[..., None, :, :], axis=-1)
    similarity = np.exp(-np.square(distance) / (2 * scale))
    return similarity


@flatten_batch(bdim=-1)
@normalize_args(euclidean=True)
def similarity_exp(X, relevant_features=None, normalize=True, scale=1.0):
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


# ---- Computing entropies for classifiers ---- #


def entropy(X, n_classes=None, edim=1, block_size=None, labels=None):
    """
    Entropy (in bits) of each "row" of `X`.

    Args:
        X: ensemble of class predictions, of shape
            ... [x ensemble_size] [x n_classes]
        edim: the ensemble dimension to average out. Default is
            1. If `None`, or not in `range(X.ndim)`, no averaging is done.
    """
    labels = get_label_type(labels, X, edim)

    if labels == Output.PROB:
        if X.ndim - 1 == edim:
            prob = np.stack((1 - X, X), axis=-1)
        elif X.shape[edim + 1] == 1:
            prob = np.concatenate((1 - X, X), axis=-1)

        if edim is None or edim >= X.ndim:
            prob = X
        else:
            prob = X.mean(axis=edim)

    else:  # integer classes
        n_classes = n_classes or _n_classes(X)
        # if X.ndim - bdim == 0:
        #    X = X[None,:]  #, numpy.newaxis]
        prob = (np.asarray(X[..., None]) == np.arange(n_classes)).astype(np.float32).mean(axis=edim)

    log2_prob = np.log2(prob)
    log2_prob[prob == 0] = 1  # When prob==0., the information limits to 0. as well.
    return -np.sum(log2_prob * prob, axis=-1)


def joint_entropy(  # NOSONAR
    X0,
    X1=None,
    n_classes=None,
    labels=None,
    block_size=None,
    indices=None,
    return_iterator=False,
    dtype=None,
    device=None,
    pbar=None,
):
    """Returns the pairwise joint entropy between elements of `X`.

    Pairing is between samples indexed in the last batch dimension. Earlier batch
    dimensions are unpaired.

    Args:
        X0 - ensemble of class predictions, of shape
            ... x batch0 x ensemble_size [x n_classes]. Entries may be
            floating point (probabilities for each class) or integer/boolean
            (indexing classes, so one fewer axes).

            If X0 has just 2 axes *and* is floating point,
            it is assumed to represent probabilities for the positive case
            of a binary classifiation. If you want to represent a
            binary classification probability, but with more batch dimensions,
            the last axis must be of size 2.

        X1 - Another ensemble of class predictions, defaults to None. If X1 is not
            None, the joint entropy between rows of X0 and rows of X1 is computed.
            This is used in, eg., doing a chunked computation for a large batch.
            If X1 is None, we take X1 = X0.
    Returns:
        batch0 x batch1
    """
    diag = X1 is None
    if device is None:
        device = np.device(X0)
    X0 = np.to(X0, device)
    X1 = X0 if diag else np.to(X1, device)
    n_0, n_1 = len(X0), len(X1)

    labels = get_label_type(labels, X0)
    if not n_classes:
        n_classes = _n_classes(X0, labels=labels, edim=1)

    if labels == Output.PROB:
        if X0.ndim == 2:
            X0 = np.stack((1 - X0, X0), axis=-1)
            X1 = np.stack((1 - X1, X1), axis=-1) if not diag else X0
        elif X0.shape[-1] == 1:
            X0 = np.concatenate((1 - X0, X0), axis=-1)
            X1 = np.concatenate((1 - X1, X1), axis=-1) if not diag else X0

    elif labels == Output.CLASS:
        if X0.shape[-1] == 1:
            X0 = X0.squeeze(-1)
            X1 = X1.squeeze(-1) if not diag else X0

    if block_size is None:
        from . import default_compute_block_size

        block_size = default_compute_block_size

    if block_size:  # preds.ndim >= 2
        assert X1 is X0, "block computation doesn't yet work between two sets of variables"
        preds = X0

        blocks0 = ranges(n_0, block_size)
        blocks1 = blocks0 if diag else ranges(n_1, block_size)
        if indices is None:
            indices = zip(*np.tril_indices(len(blocks0), 0)) if diag else np.ndindex(len(blocks0), len(blocks1))

        if isinstance(indices, Iterator):
            indices = list(indices)

        def generator_je():
            for i, j in indices:
                yield joint_entropy(
                    preds[slice(*blocks0[i])],
                    preds[slice(*blocks1[j])],
                    n_classes=n_classes,
                    labels=labels,
                    # edim=1,
                    dtype=dtype,
                    pbar=False,
                    device=device,
                    block_size=0,
                )

        blocks = generator_je()

        if return_iterator:
            return blocks

        if (pbar is None and bool(block_size)) or pbar:
            from tqdm import tqdm

            pbar = tqdm(total=int(len(blocks0) * (len(blocks0) + 1) / 2) if diag else len(blocks0) * len(blocks1))

        jen = np.zeros((n_0, n_1), dtype=dtype or np.float32, device=device)

        for block, (i, j) in zip(blocks, indices):
            jen[slice(*blocks0[i]), slice(*blocks1[j])] = block
            if i != j:
                jen[slice(*blocks0[j]), slice(*blocks1[i])] = block.T
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return jen

    # probabilities
    if labels == Output.PROB:
        # X.shape == (batch, ensemble, classes)

        # We assume that within each observation of the ensemble,
        # the given probabilities are independent between rows.
        # This will tend to underestimate the correlations.
        # Is there a better alternative (other than just using class labels instead)?
        # TODO: YES THERE IS
        x_x = X0[..., :, None, :, :, None] * X1[..., None, :, :, None, :]
        # XX.shape == (batch, batch, ensemble, n_classes**2)

        x_x = x_x.reshape(*x_x.shape[:-2], X0.shape[-1] ** 2)

    # classes
    elif labels == Output.CLASS:  # np.is_integer(X0):
        x_x = n_classes * X0[:, None, :] + X1[None, :, :]

    else:
        raise ValueError("Labels must be either 'prob' or 'class'")

    return entropy(x_x, n_classes=n_classes**2, labels=labels, edim=2, block_size=block_size)


def mutual_info(X, n_classes=None, block_size=None, **kwargs):
    entropy_2x2 = joint_entropy(X, n_classes=n_classes, block_size=block_size, **kwargs)
    entropy_1x1 = 0.5 * np.diagonal(entropy_2x2)

    return entropy_1x1[..., :, None] + entropy_1x1[..., None, :] - entropy_2x2


def approx_batch_entropy(X, **kwargs):
    """
    Approximates the joint entropy (information content) of batches of arbitrary size.
    The last batch dimension indexes the samples within each batch. Earlier batch
    dimensions index different batches.
    """
    jen = joint_entropy(X, **kwargs)
    n_samples = jen.shape[-1]
    return jen[..., np.tril_indices(n_samples, -1)].sum(axis=-1) / (n_samples - 1)


def get_label_type(labels, X, edim=1):
    if isinstance(labels, Output):
        return labels
    if labels is None:
        return Output.PROB if (np.is_float(X) or (np.is_bool(X) and X.ndim > edim + 1)) else Output.CLASS
    elif isinstance(labels, str):
        return get_output_type(labels)


def _n_classes(X, labels=None, edim=None):
    if np.is_float(X):
        if edim == X.ndim:
            return 2
        return X.shape[-1]
    return int(np.max(X) + 1)  # classes should be [0;NUM_CLASSES-1]


# -------------------------------------------------- #


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
