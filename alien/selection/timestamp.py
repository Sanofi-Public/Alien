import numpy as np

from ..decorators import get_defaults_from_self
from .selector import SampleSelector


class TimestampSelector(SampleSelector):
    def __init__(self, model=None, random_seed=None, timestamp_key="t", timestamps=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.timestamp_key = timestamp_key
        self.timestamps = timestamps

    @get_defaults_from_self
    def _select(
        self,
        samples=None,
        batch_size=None,
        timestamps=None,
        full_samples=None,
        timestamp_key=None,
        **kwargs
    ):
        if timestamps is None:
            assert hasattr(
                full_samples, timestamp_key
            ), "Must provide either timestamps, or timestamp key into the metadata"
            timestamps = getattr(full_samples, timestamp_key)

        # Gives an index from the timestamps array into the sorted array
        # of unique timestamps.
        bin = np.unique(timestamps, return_inverse=True)[1].astype(float)

        # Add a small random perturbation to the unique indices, to
        # randomize the sort order within each equal-time bin
        bin += np.random.default_rng(self.random_seed).uniform(-0.1, 0.1, len(bin))

        return np.argsort(bin)[:batch_size]
