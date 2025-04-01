from ..decorators import get_defaults_from_self
from ..models import Classifier, Output, Regressor
from ..tumpy import tumpy as tp
from ..utils import chunks, concatenate, softmax
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

    Other parameters are described in :class:`SampleSelector`.
    """

    def __init__(self, model=None, samples=None, num_samples=None, gamma=1, oversample=2, **kwargs):
        super().__init__(
            model=model,
            samples=samples,
            num_samples=num_samples,
            **kwargs,
        )
        if isinstance(model, Classifier):
            self.grad_embedding = exp_grad_embedding.__get__(self)
        elif isinstance(model, Regressor):
            self.grad_embedding = mse_grad_embedding.__get__(self)
        self.gamma = gamma
        self.oversample = oversample

    @get_defaults_from_self
    def _select(  # NOSONAR
        self, batch_size=None, samples=None, labelled_samples=None, fixed_samples=None, verbose=None, **kwargs
    ):
        if labelled_samples is None or len(labelled_samples) == 0:
            return self.rng.choice(len(samples), batch_size, replace=False)

        with tp.no_grad():

            if getattr(labelled_samples, "has_Xy", False):
                labelled_samples = labelled_samples.X
            labelled_samples = concatenate(labelled_samples, fixed_samples)

            V_u = self.grad_embedding(samples, output_type=getattr(self.model, "wrapped_output", None))
            d, K = V_u.shape[1:3]  # size of penultimate layer, and number of classes
            V_u = V_u.reshape((V_u.shape[0], -1, V_u.shape[-1]))

            V_l = self.grad_embedding(
                labelled_samples, output_type=getattr(self.model, "wrapped_output", None)
            ).reshape((len(labelled_samples), -1, V_u.shape[-1]))
            V = tp.concatenate((V_l, V_u), axis=0)

            F_u = fisher_from_grad_embeddings(V_u, batch_total_size=self.compute_batch_size, normalize=False)
            F_l = fisher_from_grad_embeddings(V_l, batch_total_size=self.compute_batch_size, normalize=False)
            F = F_u + F_l

            ind_s = []  # selected indices

            # current inverse matrix
            M_inv = tp.linalg.inv(self.gamma * tp.eye(K * d) + F_l)

            # greedy forward sampling
            if verbose:
                print("Greedy forward sampling...", flush=True)
            for _ in range(min(int(batch_size * self.oversample), len(V_u))):
                # print(f"    ... {_}/{min(int(batch_size * self.oversample), len(V_u))}", flush=True)
                A = V_u.swapaxes(1, 2) @ M_inv @ V_u + tp.eye(K)
                if A.shape[-1] == 1:
                    A_inv = 1 / A
                else:
                    A_inv = tp.linalg.pinv(A)
                score = tp.trace(V_u.swapaxes(1, 2) @ M_inv @ F @ M_inv @ V_u @ A_inv, axis1=-2, axis2=-1)

                for ind in tp.flip(tp.argsort(score)):
                    i = ind if isinstance(ind, int) else ind.item()
                    if i not in ind_s:
                        ind_s.append(i)

                        V_i = V_u[i]
                        M_inv -= M_inv @ V_i @ A_inv[i] @ V_i.swapaxes(-2, -1) @ M_inv
                        break

            # greedy backwards pruning
            if verbose:
                print("Greedy backward pruning...", flush=True)
            for _ in range(len(ind_s) - batch_size):
                # print(f"    ... {len(ind_s) - batch_size - _}/{len(ind_s) - batch_size}", flush=True)

                V_s = V_u[ind_s]  # selected samples

                A = V_s.swapaxes(1, 2) @ M_inv @ V_s - tp.eye(K)
                if A.shape[-1] == 1:
                    A_inv = 1 / A
                else:
                    A_inv = tp.linalg.pinv(A)
                score = tp.trace(V_s.swapaxes(1, 2) @ M_inv @ F @ M_inv @ V_s @ A_inv, axis1=-2, axis2=-1)

                ind = tp.argmax(score)
                del ind_s[ind]

                V_i = V_s[ind]
                M_inv -= M_inv @ V_i @ A_inv[ind] @ V_i.swapaxes(-2, -1) @ M_inv

        return ind_s


def fisher_cross_entropy(
    model,
    X,
    output_type="logit",
    batched=False,
    batch_size=None,
    batch_total_size=None,
    return_embedding=True,
    normalize=False,
):
    """Compute the integrated (summed) Fisher information matrix for the given classifier on the given input.

    :param X: The input data.
    :param model: The model to use. If `model.predict_with_embedding` exists, call that. Otherwise, call `model`
        directly. In either case, must return `(output, embedding)`,
        where `embedding` is the activations in the last hidden layer. Assumes a softmax for the probabilities and
        cross-entropy loss.
    :param output_type: The type of output to use. Can be 'logit' or 'prob'.


    :return: The Fisher information matrix.
    """
    V = exp_grad_embedding(model, X, output_type=output_type)
    F = fisher_from_grad_embeddings(
        V, batched=batched, batch_size=batch_size, batch_total_size=batch_total_size, normalize=normalize
    )

    return F, V if return_embedding else F


def fisher_from_grad_embeddings(V, batched=False, batch_size=None, batch_total_size=None, normalize=False):
    """
    Compute the Fisher information matrix from the gradient embeddings.


    """
    if V.ndim == 4:
        V = V.reshape((len(V), -1, V.shape[-1]))
    K = V.shape[-1]
    d = V.shape[-2] // K

    if batched and not (batch_size or batch_total_size):
        batch_total_size = 1e8  # Not likely to cause memory issues

    if batch_total_size:
        batch_size = int(batch_total_size // (K * d) ** 2)  # d * d * k * k

    F = sum((v @ v.swapaxes(-1, -2)).sum(0) for v in chunks(V, batch_size))
    return F if not normalize else F / len(V)


def exp_grad_embedding(model, X, output_type="logit"):
    """Compute the ...

    :param X: The input data.
    :param model: The model to use. Must implement `predict_with_embedding`, which returns `(output, embedding)`,
        where `embedding` is the activations in the last hidden layer. Assumes a softmax for the probabilities and
        cross-entropy loss.
    :param output_type: The type of output to use. Can be 'logit' or 'prob'.

    :return: ... shape: (N, d, K, K)
    """
    if isinstance(model, SampleSelector):
        model = model.model
    out, emb = getattr(model, "predict_with_embedding", model)(X)
    p = softmax(out, axis=-1) if Output(output_type) == Output.LOGIT else out

    return emb[..., None, None] * (tp.eye(p.shape[-1]) - p[..., None, None, :]) * tp.sqrt(p[..., None, :, None])
    # in (eye - p), the probabilities vary along the last axis,
    # in sqrt(p), they vary along the second-to-last axis


def mse_grad_embedding(model, X, output_type=None):
    """
    For regression with MSE loss, the gradient embeddings are equivalent to
    the penultimate layer embeddings.

    :param X: The input data.
    :param model: The model. Must implement `predict_with_embedding`, or `embedding`, or if neither,
        `model` is called directly. Must return either `(output, embedding)` or `embedding`.
    :param output_type: Ignored.

    :return: The gradient embeddings.
    """
    if isinstance(model, SampleSelector):
        model = model.model
    if hasattr(model, "predict_with_embedding"):
        _, emb = model.predict_with_embedding(X)
    elif hasattr(model, "embedding"):
        emb = model.embedding(X)
    else:
        emb = model(X)
    return emb[..., None, None]
