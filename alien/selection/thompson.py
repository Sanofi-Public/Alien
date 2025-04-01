import numpy as np
from .selector import SampleSelector


class ThompsonSelector(SampleSelector):
    """
    Implements Thompson sampling, a form of Bayesian optimization.
    Each element of the batch is chosen from the sample pool according
    to its probability of being the highest scorer. Equivalently, to
    select a single element of the batch, we sample the model from some
    posterior over model parameters, then use this sampled model to
    give correlated predictions over the whole sample pool, and select
    the sample with the highest prediction in that case.

    Args:
        sign: If `True` or any positive number, seeks the
            maximum value. Otherwise, seeks the minimum value.
            Defaults to +1.
        multiple: The distribution of predictions
            is modified so that the covariances are scaled by a factor of
            `multiple`. A number > 0.

            This can be regarded as a parameter tuning the explore-vs.-exploit
            emphasis, with smaller numbers being greedier/more exploitative.
            Most of our methods of computing (co)variance systematically
            underestimate the magnitude, which means that (without a `multiple`
            factor) we will underestimate how likely it is for a sample with
            a low predicted value to come out on top as the winner. In the
            limit where uncertainties are (wrongly) estimated to be 0,
            Thompson sampling reduces to greedy selection, with all of its
            problems. Therefore, to get good performance out of this, you
            will likely have to experiment with values larger than 1.0.
    """

    def __init__(
        self,
        model=None,
        samples=None,
        num_samples=float("inf"),
        batch_size=1,
        multiple=1.0,
        sign=+1,
        **kwargs
    ):
        super().__init__(
            model=model, samples=samples, num_samples=num_samples, batch_size=batch_size, **kwargs
        )
        self.multiple = multiple
        self.sign = +1 if sign == True or sign > 0 else -1

    def _select(self, samples=None, batch_size=None, multiple=1.0, **kwargs):
        predict_args = {"multiple": multiple} if multiple != 1 else {}
        ensemble = self.sign * self.model.predict_samples(samples, n=batch_size, **predict_args)
        return np.argmax(ensemble, axis=0)
