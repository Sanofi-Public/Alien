import numpy as np

from ..decorators import get_defaults_from_self
from .selector import SampleSelector


class RandomSelector(SampleSelector):
    """Select samples at random."""

    def __init__(self, model=None, random_seed=None, **kwargs):
        super().__init__(model=model, **kwargs)

    @get_defaults_from_self
    def _select(self, samples=None, batch_size=1, **kwargs):
        return self.rng.choice(
            len(samples), batch_size, replace=False
        )
