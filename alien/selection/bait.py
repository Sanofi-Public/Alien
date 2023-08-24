import numpy as np

from ..data import Dataset
from ..decorators import get_defaults_from_self
from ..utils import concatenate, dot_last
from .selector import SampleSelector


class BAITSelector(SampleSelector):
    """
    Batch selector following the BAIT strategy. See ` <https://arxiv.org/abs/2106.09675>`_.
    This strategy optimizes the trace of the Fisher matrix between the outputs
    and the last layer of parameters. This is a measure of the mutual information between
    the unknown labels and the parameters.

    BAIT optimizes the trace of the Fisher for the "batch" consisting of all previously
    labelled samples plus the unlabelled candidate samples. This means that BAITSelector
    needs to know the previously labelled samples. They can be passed into either :meth:`__init__`
    or :meth:`select`, as `labelelled_samples`. (This class will try to determine whether
    `labelled_samples` needs to be unpacked into separate X and y columns---only the X 
    column is needed.)

    There are two hyperparameters, `gamma` and `oversample`, described below.

    :param model: An instance of models.LinearizableRegressor, or a model which
        implements the `embedding` method.
    :param samples: The sample pool to select from. Can be a numpy-style addressable
        array (with first dimension indexing samples, and other dimensions indexing
        features)---note that alien.data.Dataset serves this purpose---or an instance of
        sample_generation.SampleGenerator, in which case the num_samples parameter is
        in effect.
    :param num_samples: If a `SampleGenerator` has been provided via the 'samples'
        parameter, then at the start of a call to self.select(...), `num_samples`
        samples will be drawn from the `SampleGenerator`, or as many samples as the
        `SampleGenerator` can provide, whichever is less. Defaults to Inf, i.e., draws
        as many samples as available.
    :param labelelled_samples: The samples which have already been labelled (or are in the
        process of being labelled). This class will try to determine whether
        `labelled_samples` needs to be unpacked into separate X and y columns---only the X 
        column is needed.
    :param batch_size: Size of the batch to select.
    :param random_seed: A random seed for deterministic behaviour.
    :param gamma: The 'regularization' parameter in the BAIT algorithm. A larger value
        corresponds to narrower priors. Defaults to 1, which works well enough.
    :param oversample: The factor by which to oversample in the greedy acquisition step.
        BAIT will greedily draw a batch of `oversample * batch_size` samples, then
        greedily remove all but `batch_size` of them. Defaults to 2, which is empirically
        good.
    """

    def __init__(
        self,
        model=None,
        samples=None,
        num_samples=None,
        gamma=1,
        oversample=2,
        random_seed=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            samples=samples,
            num_samples=num_samples,
            **kwargs,
        )
        self.gamma = gamma
        self.oversample = oversample
        self.rng = np.random.default_rng(random_seed)

    @get_defaults_from_self
    def _select(
        self,
        batch_size=None,
        samples=None,
        labelled_samples=None,
        fixed_samples=None,
        verbose=None,
        **kwargs
    ):
        if labelled_samples is None or len(labelled_samples) == 0:
            return self.rng.choice(len(samples), batch_size, replace=False)

        if getattr(labelled_samples, 'has_Xy', False):
            labelled_samples = labelled_samples.X
        labelled_samples = concatenate(labelled_samples, fixed_samples)

        X_u = np.asarray(self.model.embedding(samples))
        X_l = np.asarray(self.model.embedding(labelled_samples))
        X = np.concatenate((X_l, X_u), axis=0)

        emb_dim = X.shape[-1]  # size of embedding space

        ind_s = []  # selected indices

        # Fisher matrix for the whole universe
        F = (X[..., :, None] * X[..., None, :]).sum(axis=0)

        # Fisher matrix for labelled samples only
        F_l = (X_l[..., :, None] * X_l[..., None, :]).sum(axis=0)

        # current inverse matrix
        M_inv = np.linalg.inv(self.gamma * np.eye(emb_dim) + F_l)

        # greedy forward sampling
        if verbose:
            print("Greedy forward sampling...", flush=True)
        for _ in range(min(int(batch_size * self.oversample), len(X_u))):
            A = dot_last(X_u, np.dot(X_u, M_inv)) + 1
            A[A == 0] = np.finfo("float32").tiny

            score = dot_last(X_u, np.dot(X_u, M_inv @ F @ M_inv)) / A

            for ind in np.argsort(score)[::-1]:
                if ind not in ind_s:
                    ind_s.append(ind)

                    X_i = X_u[ind]
                    M_inv -= M_inv @ (X_i[..., :, None] * X_i[..., None, :]) @ M_inv / A[ind]
                    break

        # greedy backwards pruning
        if verbose:
            print("Greedy backward pruning...", flush=True)
        for _ in range(len(ind_s) - batch_size):
            X_s = X_u[ind_s]  # selected samples

            A = dot_last(X_s, np.dot(X_s, M_inv)) - 1

            score = dot_last(X_s, np.dot(X_s, M_inv @ F @ M_inv)) / A

            ind = np.argmax(score)
            del ind_s[ind]

            X_i = X_s[ind]
            M_inv -= M_inv @ (X_i[..., :, None] * X_i[..., None, :]) @ M_inv / A[ind]

        return ind_s
