from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Optional

from ...utils import dict_pop
from ...config import default_training_epochs, default_training_samples

limit_long_names = ["sample_limit", "batch_limit", "epoch_limit"]
limit_short_names = ["samples", "batches", "epochs"]
limit_name_pairs = list(zip(limit_long_names, limit_short_names))
# limit_short_to_long = dict(zip(limit_short_names, limit_long_names))
limit_long_to_short = dict(limit_name_pairs)


# decorator
def get_training_limit(fn):
    """
    *** Decorator ***

    Modifies a function so that it can take any of a number of
    different naive arguments to specify limits to pytorch training.
    The inner (wrapped) function will see only an argument
    `training_limit`, which receives an instance of the fancy
    TrainingLimit class.

    Thus, the inner fn must have an argument named `training_limit`
    (or `**kwargs`).

    The outer, decorated fn may take additional optional kwargs:
        `'sample_limit'`, `'batch_limit'`, `'epoch_limit'`
        `'samples'`,      `'batches'`,     `'epochs'`
    If these are given, they determine the value of training_limit
    according a calculation that favors them in the order
    provided (though you may only want to provide one).
    """

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        lim_kwargs = dict_pop(kwargs, *limit_name_pairs)

        if "training_limit" not in kwargs:
            if len(lim_kwargs) > 0:
                kwargs["training_limit"] = StdLimit(**lim_kwargs)

        return fn(*args, **kwargs)

    return wrapped_fn


@dataclass
class TrainingLimit:
    """
    Encapsulates the computation of training limits, which may depend on
    things like dataset length.
    """

    min_samples: int = 0
    min_epochs: float = 0

    samples: Optional[int] = None
    epochs: Optional[float]   = None
    batches: Optional[int] = None

    max_samples: float = float("inf")
    max_epochs: float = float("inf")

    def sample_limit(self, length=None):
        if length is None:
            min_samples = self.min_samples
            max_samples = self.max_samples
            samples = self.samples
        else:
            min_samples = max(self.min_samples, self.min_epochs * length)
            max_samples = min(self.max_samples, self.max_epochs * length)
            samples = self.epochs * length if self.epochs is not None else self.samples

        if min_samples > max_samples:
            return 0.5 * (min_samples + max_samples)
        elif samples is None:
            if max_samples == float("inf"):
                return min_samples if min_samples > 0 else \
                    (default_training_epochs * length if length else \
                    default_training_samples)
            return 0.5 * (min_samples + max_samples)

        if samples < min_samples:
            return min_samples
        elif samples > max_samples:
            return max_samples
        else:
            return samples

    def batch_limit(self, batch_size=None, length=None):
        if self.batches:
            return self.batches
        if not batch_size:
            raise ValueError(
                "Must provide positive batch_size to method batch_limit(...)\n   Or else pass batch_limit into the constructor."
            )
        return self.sample_limit(length=length) // batch_size


class StdLimit(TrainingLimit):
    def __init__(self, **kwargs):
        assert set(kwargs).issubset(
            limit_long_names
        ), f"For StdLimit, may only pass in\n{*limit_long_names, *limit_short_names} \nYou may want to try TrainingLimit."
        kwargs = {limit_long_to_short[name]: v for name, v in kwargs.items()}
        super().__init__(**kwargs)


default_limit = TrainingLimit(min_samples=1e4, min_epochs=10)
