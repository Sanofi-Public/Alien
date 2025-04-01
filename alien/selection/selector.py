"""Base class for selecting samples and helper functions."""

import math
import sys
from abc import ABCMeta, abstractmethod
from numbers import Number

from ..classes import final
from ..data import DictDataset
from ..decorators import get_defaults_from_self
from ..sample_generation import SampleGenerator
from ..stats import augment_ensemble, rankdata
from ..tumpy import tumpy as np
from ..utils import chunks, concatenate, isint


class SampleSelector(metaclass=ABCMeta):
    """
    Abstract base class for selection strategies.

    Args:
        model (models.Model): Will be used to sample predictions
        samples: The sample pool to select from. Can be a numpy-style addressable
            array (with first dimension indexing samples, and other dimensions indexing
            features)---note that :class:`alien.data.Dataset` serves this purpose---or an instance of
            :class:`sample_generation.SampleGenerator`, in which case the num_samples
            parameter is in effect.
        labelled_samples: Some selection strategies need to know the previously-labelled
            samples.
        batch_size: Size of the batch to select.
        num_samples: If a `SampleGenerator` has been provided via the 'samples'
            parameter, then at the start of a call to :meth:`.select`, `num_samples`
            samples will be drawn from the `SampleGenerator`, or as many samples as the
            `SampleGenerator` can provide, whichever is less. Defaults to Inf, i.e., draws
            as many samples as available.
        prior: Specifies a "prior probability" for each sample. Each selector
            may use this prior as it sees fit, but generally, samples with low
            prior are de-emphasized in the selection process. This is a covenient way of
            introducing factors other than uncertainty into the ranking. `prior` may be an
            array of numbers
            (of size num_samples), or a function (applied to the samples), or one of the
            following:
                'prediction': calculates a prior from the quantile of the predicted
                    performance (not the uncertainties). `prior_scale` sets the power
                    this quantile is raised to.
            Defaults to the constant value 1.
        prior_scale: The prior will be raised to this power before applying it to
            the samples. Defaults to 1.
        prefilter: Reduces the incoming sample pool before applying batch
            selection. If `a` is the single-sample acquisition function, then
            `prefilter = True` selects a subset of the provided samples maximizing
            `a`

            If 0 < prefilter < 1, takes this fraction of the
            sample pool. If prefilter >= 1, takes this many samples.

            Some of the selectors are limited in how many samples they can consider
            for the final, batch-selection problem. For example, :class:`CovarianceSelector`
            computes and stores the size-N^2 covariance matrix for the whole
            sample pool; therefore, because of memory constraints it should work with
            at most around 10,000 samples.

            In such cases, there is often a cheaper prefiltering operation available.
            Eg., :class:`CovarianceSelector` prefilters only with the variance, rather
            than the full covariance.

            A practical strategy in such cases is to take a sample pool about
            5 times as big as the selector can handle for the final computation, then
            narrow down to only the top 20% individual scores before batch selection.
            Narrowing to much less than 20% risks reducing diversity too much and
            changing what would ultimately be the selected batch.
        random_seed: A random seed for deterministic behaviour.
        compute_batch_size: When doing computations, try to keep tensors to this
            total number of elements. Adjusting this can save you from memory errors.
        return_indices: If True, :meth:`.select` will return the indices of
            the selection (from within the given sample pool). If False,
            :meth:`.select` will return the actual selected samples. Defaults to `False`.

    """

    def __init__(
        self,
        model=None,
        batch_size=1,
        samples=None,
        num_samples=None,
        labelled_samples=None,
        X_key="X",
        prior=None,
        prior_scale=1,
        random_seed=None,
        return_indices=False,
        verbose=1,
        prefilter=None,
        compute_batch_size=1e8,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.labelled_samples = labelled_samples
        self.samples = samples
        self.num_samples = num_samples
        self.X_key = X_key
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.compute_batch_size = compute_batch_size
        if prior == "prediction":
            self.prior_func = self.prediction_prior
        elif callable(prior):
            self.prior_func = prior
        else:
            import numpy

            array_types = (numpy.ndarray,)
            if "torch" in sys.modules:
                import torch

                array_types += (torch.Tensor,)
            if isinstance(prior, array_types):
                self.prior_func = lambda X: prior
            else:
                self.prior_func = None
        if prior_scale != 1:
            self.old_prior_func = self.prior_func
            self.prior_func = lambda X: np.power(self.old_prior_func(X), prior_scale)

        self._last_pred = None
        self._last_std = None
        self._last_X = None
        self.return_indices = return_indices
        self.verbose = verbose
        self.prefilter = prefilter

    def prediction_prior(self, samples):
        # TODO: Maybe this should use self.samples if samples is not given
        pred = self.model_predict(samples)
        return rankdata(pred) / len(np.ravel(pred))

    def model_predict(self, X):
        """
        Convenience function to be used in place of `model.predict`. Caches the last predicted
        value, and returns the cached value if `X is self._last_X`. Saves compute time if
        there are lots of conditional executions of `model.predict`.
        """
        if X is not self._last_X:
            self._last_pred = self.model.predict(X)
            self._last_X = X
        return self._last_pred

    def get_prefilter(self, X=None, k=None, prior=1, score=None, return_indices=True):
        if k is None:
            k = self.prefilter
        if k is None or k == 1:
            return np.arange(len(X)) if return_indices else X

        if score is None:
            if X is None:
                raise ValueError("Must provide either 'X' or 'score' to the method 'get_prefilter'")
            score = self.prefilter_score(X)

        if 0 < k < 1:
            k = int(round(k * len(score)))
        else:
            k = min(k, len(score))

        # adjust score by prior
        score *= prior

        indices = np.argsort(score)[-int(k) :]
        return indices if return_indices else X[indices]

    def prefilter_score(self, X):
        assert hasattr(
            self.model, "uncertainty"
        ), "You have provided no `prefilter_score`, nor does your model have an `uncertainty` method"
        if X is not self._last_X:
            self._last_uncertainty = self.model.uncertainty(X)
            self._last_X = X
        return self._last_uncertainty

    @final
    @get_defaults_from_self
    def select(
        self,
        batch_size=None,
        samples=None,
        num_samples=None,
        prior=None,
        X_key=None,
        fixed_samples=None,
        fixed_prior=None,
        prefilter=None,
        return_indices=None,
        **kwargs,
    ):
        """
        Selects a batch from the provided samples, and returns it (or its
        indices).

        All of the arguments to :meth:`.select` are optional. If you have provided
        `samples` and other necessary parameters to the constructor already,
        then you may omit them here.

        However, some parameters here are *not* in the constructor: `fixed_samples` and
        `fixed_prior`.

        Args:
            batch_size (int, optional):
                The size of the batch to select.
            samples (ArrayLike, optional):
                The sample pool to select from. Can be a numpy-style addressable
                array (with first dimension indexing samples, and other dimensions indexing
                features)---note that :class:`alien.data.Dataset` serves this purpose---or an instance of
                :class:`sample_generation.SampleGenerator`, in which case the num_samples
                parameter isnin effect.
            num_samples
                If a `SampleGenerator` has been provided via the 'samples'
                parameter, then at the start of a call to :meth:`.select`, `num_samples`
                samples will be drawn from the `SampleGenerator`, or as many samples as the
                `SampleGenerator` can provide, whichever is less. Defaults to Inf, i.e., draws
                as many samples as available.

            prior
                A "prior probability" for each sample. May be an array of numbers, a function,
                or the string `'prediction'`. A more detailed explanation is above,
                in the class definition. Defaults to the constant value 1.
            prior_scale
                The prior will be raised to this power before applying it to
                the samples. Defaults to 1.
            prefilter
                Reduces the incoming sample pool before applying batch
                selection. If 0 < prefilter < 1, we use this fraction of the
                sample pool. If prefilter >= 1, we use this many samples. A more detailed explanation
                if above, in the class definition.
            return_indices
                If True, :meth:`.select` will return the indices of
                the selection (from within the given sample pool). If False,
                :meth:`.select` will return the actual selected samples. Defaults to `False`.

            X_key
                The key used to extract the X values from `samples`. I.e.,
                `X = samples[X_key]`. This is only in effect if you pass an
                explicit value to `X_key`, or if `samples` is a :class:`DictDatabase`
                with key `'X'`. By default, `X = samples`.

            fixed_samples (ArrayLike, optional)
                This parameter is for passing in those samples which have been
                previously selected for labeling, but which haven't been labeled yet.
                (Eg., you've previously sent off a batch to the laboratory pipeline for
                testing, but you need to select the next batch for the pipeline before
                the results are in.)
                Some selection strategies (eg., :class:`CovarianceSelector` and
                :class:`BAITSelector`) will use this information to avoid redundancy
                between the newly selected batch and the `fixed_samples`.
            fixed_prior:
                If you provide an explicit (i.e., array-like) prior for `samples`, then
                you must also provide a prior for `fixed_samples`.


        Returns:
            The selected batch, either as a sub-array of `samples`, or as an array of
            indices into `samples` (if `return_indices` is set to True).
        """
        self._last_X = None

        # If 'samples' is a SampleGenerator, generate a pool of samples
        if isinstance(samples, SampleGenerator):
            samples = samples.generate_samples(float("inf") if num_samples is None else num_samples)

        # If `samples` has a key or attribute `X_key`, pull this X out of
        # samples
        if isinstance(samples, DictDataset) and hasattr(samples, X_key):
            kwargs["full_samples"] = samples
            X = getattr(samples, X_key)
            if fixed_samples is not None:
                kwargs["full_fixed_samples"] = fixed_samples
                fixed_X = getattr(fixed_samples, X_key)
            else:
                fixed_X = None
        else:
            X, fixed_X = samples, fixed_samples

        prior, fixed_prior = self._get_prior(fixed_X, X, prior)

        if prefilter not in {None, 1}:
            pre_indices = self.get_prefilter(X, prefilter, prior=prior, return_indices=True, **kwargs)
            X, prior = (X[pre_indices], prior[pre_indices]) if X is not None else (pre_indices, prior)

        indices = self._select(
            batch_size=batch_size,
            samples=X,
            fixed_samples=fixed_X,
            prior=prior,
            fixed_prior=fixed_prior,
            **kwargs,
        )

        return indices if (return_indices or X is None) else samples[indices]

    def _get_prior(self, fixed_X, X, prior=None):
        if prior == "prediction":
            prior = None
            self.prior_func = self.prediction_prior

        # prior is unspecified, get it by calling self.prior_func
        if prior is None:
            if self.prior_func is not None:
                if fixed_X is not None:
                    prior = self.prior_func(concatenate(fixed_X, X))
                    fixed_prior, prior = prior[: len(fixed_X)], prior[len(fixed_X) :]
                else:
                    prior = self.prior_func(X)
            else:
                prior = 1
                fixed_prior = 1
        if isinstance(prior, Number):
            prior = np.asarray([prior])
        if X is not None:
            prior = np.broadcast_to(prior, len(X))
        return prior, fixed_prior

    @abstractmethod
    def _select(self, batch_size=None, samples=None, prior=None, **kwargs):
        pass


# @profile
def optimize_batch(  # NOSONAR
    scoring_fn,
    batch_size,
    samples,
    n_fixed=0,
    scoring_opt_step=None,
    n_tuples=None,
    n_rounds=None,
    n_starts=None,
    parallel_starts=False,
    random_seed=None,
    callback=None,
    verbose=1,
    scoring_capacity=None,
):
    if isint(samples):
        samples = np.arange(samples, dtype=np.int64)
    n_samples = len(samples)

    n_tuples = n_tuples or max(100, int(math.sqrt(n_samples)))
    n_starts = min(n_starts or 10, n_samples, n_tuples)
    n_rounds = n_rounds or 20

    rng = np.random.default_rng(random_seed)

    # build distribution using singleton scores
    if verbose:
        print("Computing singleton scores...")
    single = scoring_fn(samples[:, None, ...]).reshape((-1,))

    if batch_size == 1:
        return samples[[np.argmax(single.flatten())]]

    tuples = _generate_initial_batches(samples, n_tuples, batch_size, single, rng, verbose=verbose)
    assert tuples.shape == (n_tuples, batch_size)

    # if we have fixed samples, put them in at the front of every batch
    if n_fixed > 0:
        fixed_batch = np.broadcast_to(samples[:n_fixed], (n_tuples, n_fixed))
        tuples = np.concatenate([fixed_batch, tuples], axis=-1)
        batch_size += n_fixed

    if verbose:
        print("Scoring initial batches...")
    # scoring fn should eat 2nd axis (batch_size)
    if scoring_capacity is not None:
        scores = np.concatenate([scoring_fn(t) for t in chunks(tuples, scoring_capacity)])
    else:
        scores = scoring_fn(tuples)

    best_tuples = tuples[np.argsort(scores)[-n_starts:]]
    assert best_tuples.shape == (n_starts, batch_size)

    if isinstance(parallel_starts, bool):
        parallel_starts = n_starts if parallel_starts else 1
    parallel_starts = min(parallel_starts or 1, n_starts)

    if verbose:
        print("Greedy optimization of best batches...")
    # Greedy optimization of best_tuples
    best_scores = np.full(n_starts, -np.inf)

    if parallel_starts < n_starts:
        parallel_starts = chunks(n_starts, parallel_starts)
        for start in parallel_starts:
            if len(start) > 1:
                print(f"Running multi-starts [{start[0]+1}..{start[-1]+1}]...")
            else:
                print(f"Running multi-start [{start[0]+1}]...")
            best_tuples[start], best_scores[start] = _optimize_loop(
                best_tuples=best_tuples[start],
                n_rounds=n_rounds,
                n_fixed=n_fixed,
                samples=samples,
                scoring_fn=scoring_fn,
                scoring_opt_step=scoring_opt_step,
                verbose=verbose,
            )
    else:
        best_tuples, best_scores = _optimize_loop(
            best_tuples=best_tuples,
            n_rounds=n_rounds,
            n_fixed=n_fixed,
            samples=samples,
            scoring_fn=scoring_fn,
            scoring_opt_step=scoring_opt_step,
            verbose=verbose,
        )

    return best_tuples[np.argmax(best_scores)][n_fixed:]


def _optimize_loop(
    best_tuples,
    n_rounds,
    n_fixed,
    samples,
    scoring_fn,
    scoring_opt_step,
    verbose,
    callback=None,
):
    n_starts, n_samples, batch_size = len(best_tuples), len(samples), best_tuples.shape[1]

    for round_ in range(n_rounds):
        if verbose:
            print(f"    Optimization round {round_+1} - ", end="")

        n_changed = 0

        for i in range(n_fixed, batch_size):
            if callback:
                callback(round_, i, best_tuples)
            # if n_rounds * batch_size < 100:
            #    print(f"    {i+1}: Optimizing over batch coordinate {i}...")

            scores = _scoring_step(
                i,
                best_tuples,
                samples,
                n_samples,
                n_starts,
                batch_size,
                scoring_fn,
                scoring_opt_step,
            )
            assert scores.shape == (n_starts, n_samples), (scores.shape, (n_starts, n_samples))

            # exclude selections that will repeat a sample
            # by setting their scores to -inf
            scores[np.arange(n_starts)[:, None], best_tuples[:, :i]] = -np.inf
            scores[np.arange(n_starts)[:, None], best_tuples[:, i + 1 :]] = -np.inf

            # Find max score over n_samples for each batch
            next_steps = scores.argmax(axis=1)
            assert next_steps.shape == (n_starts,)

            n_changed += (best_tuples[:, i] != samples[next_steps]).sum(axis=-1)
            best_tuples[:, i] = samples[next_steps]

        frac_changed = n_changed / (n_starts * batch_size)
        if verbose:
            print(f"Changed {(frac_changed*100):.1f} %")
        if n_changed == 0:
            break

    if callback:
        callback(round_, i, best_tuples)

    if verbose:
        print("Done...\n")

    return best_tuples, scoring_fn(best_tuples)


def _generate_initial_batches(samples, n_tuples, batch_size, single, rng, verbose=False):
    """Helper function to generate initial batches in optimize_batch."""
    # Generate a collection of tuples.
    # Probability density is a linear
    # function over the ranked sample list.
    # Somewhat favours higher ranked samples
    if verbose:
        print("Generating initial batches...")
    tuples = np.empty((n_tuples, batch_size), dtype=np.int64)
    ranked_scores = rankdata(single).astype(np.float32)
    ranked_scores /= np.sum(ranked_scores)
    for i in range(n_tuples):
        selected_sample = rng.choice(samples, batch_size, replace=False, p=ranked_scores)
        tuples[i] = selected_sample
    return tuples


def _scoring_step(i, best_tuples, samples, n_samples, n_starts, batch_size, scoring_fn, scoring_opt_step):
    """Helper function to perform a scoring step in optimize batch.
    Returns `scores`, an array of shape (n_starts, n_samples)
    """
    if scoring_opt_step is None:
        # 3d array, last index broadcast to len(samples)?
        best_tuples_alternates = np.broadcast_to(best_tuples[:, None], (n_starts, n_samples, batch_size)).copy()

        best_tuples_alternates[:, :, i] = samples[None, :]  # at one place in the batch, scan over all samples
        scores = scoring_fn(best_tuples_alternates)  # scoring_fn eats axis -1, i.e, batch_size
    else:
        # We pass to the scoring function only the fixed samples of each batch
        # (plus the whole sample space to test for the variable sample).
        # Scoring function is responsible for generating whole space of counterfactual
        # batches, and evaluating them
        scores = scoring_opt_step(np.delete(best_tuples, i, axis=-1), samples)
    return scores


# --------------------------------------


def precomputed_ensemble_score(model, samples, fn, multiple=1.0, augment_ensemble_size=-1):
    """
    :param fn: scoring function. It should take predicted
        values of shape
            (..., batch_size, ensemble_size)
        I.e., the first axes counts separate batches.
        'fn' should consume the last two axes, and return
        a set of scalar scores, one for each batch.
    """
    preds = model.predict_fixed_ensemble(samples, multiple=multiple)
    if augment_ensemble_size and augment_ensemble_size > preds.shape[-1]:
        preds = augment_ensemble(preds, augment_ensemble_size, rng=model.rng)
    del model
    del samples

    def score(indices):
        """
        Returns a set of scalar scores, one for each batch.

        :param indices: indices are of shape
                (n_batches, batch_size)
        """
        return fn(preds[indices])

    return score
