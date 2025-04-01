from functools import partial

import numpy as np
from scipy.stats import norm

from ..decorators import NEED, get_defaults_from_self
from .selector import SampleSelector, optimize_batch, precomputed_ensemble_score


def expected_improvement_(pred, std_dev, benchmark, epsilon=1e-8):
    """
    Returns the expected improvement, i.e., the expected amount by
    which this result exceeds the benchmark. So, in the expectation
    calculation, values below benchmark aren't counted. Typically,
    the benchmark is the best labelled value, or related to it.

    :param pred: the outcome (possibly batched) that the model predicts
                 for the sample
    :param std_dev: the uncertainty (quantified as standard deviation,
                    again possible batched) that the model gives for its
                    prediction
    :param benchmark: the benchmark against which expected improvement,
                      is evaluated, typically related to the best labelled
                      value
    :param epsilon: a floor for std_dev in the division step of the
                    calculation, defaults to 1e-8
    """

    # Z is the renormalized integration bound:
    Z = (pred - benchmark) / np.maximum(std_dev, epsilon)

    # TODO: Reference for this solution?
    e_i = (pred - benchmark) * norm.cdf(Z) + std_dev * norm.pdf(Z)

    return e_i


def expected_improvement(samples, model, labelled_samples, margin=0.03, multiple=1, return_predictions=False):
    """Compute the expected improvement of given samples

    Args:
        samples (_type_): _description_
        model (_type_): _description_
        labelled_samples (_type_): _description_
        margin (float, optional): _description_. Defaults to 0.03.
        multiple (int, optional): _description_. Defaults to 1.
        return_predictions (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    y = labelled_samples[:, -1]
    benchmark = np.max(y) + margin * np.std(y)
    preds, std_devs = model.predict(samples)
    if multiple != 1:
        std_devs *= multiple
    e_i = expected_improvement_(preds, std_devs, benchmark).squeeze()
    if return_predictions:
        return e_i, preds, std_devs
    else:
        return e_i


def ensemble_batch_expected_improvement(benchmark, preds):
    """
    Scoring function, as in selector.precomputed_ensemble_score
    It should take predicted values of shape
        (..., batch_size, ensemble_size)
    """
    preds = preds.max(axis=-2)
    # Now preds.shape == (..., ensemble size)
    improvement = preds - benchmark
    improvement[improvement < 0] = 0
    # Averages the improvement score over the ensemble
    return np.mean(improvement, axis=-1)


class ExpectedImprovementSelector(SampleSelector):
    def __init__(
        self,
        model=None,
        labelled_samples=None,
        samples=None,
        num_samples=None,
        batch_size=1,
        multiple=1.0,
        margin=0.03,
        random_seed=None,
        augment_ensemble=0,
    ):
        super().__init__(
            model=model,
            labelled_samples=labelled_samples,
            samples=samples,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        self.margin = margin
        self.multiple = multiple
        self.augment_ensemble = augment_ensemble

    @get_defaults_from_self
    def _select(self, samples=None, labelled_samples=NEED, y_labelled=NEED, batch_size=None, **kwargs):
        benchmark = np.max(np.asarray(y_labelled)) + self.margin * np.std(np.asarray(y_labelled))

        score = partial(ensemble_batch_expected_improvement, benchmark)
        scoring_fn = precomputed_ensemble_score(
            self.model,
            samples,
            score,
            multiple=self.multiple,
            augment_ensemble_size=self.augment_ensemble,
        )

        batch_indices = optimize_batch(
            scoring_fn,
            batch_size,
            len(samples),
            # n_fixed=n_fixed,
            random_seed=np.random.default_rng(self.random_seed).integers(1e8),
        )

        return batch_indices
