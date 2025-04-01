import numpy as np

from ..decorators import get_defaults_from_self
from .selector import SampleSelector


class GreedySelector(SampleSelector):
    @get_defaults_from_self
    def _select(self, samples=None, batch_size=None):
        preds = self.model.predict(samples)
        return np.argsort(preds)[-batch_size:]
