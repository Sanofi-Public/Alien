from typing import Union

import numpy as np

from ..utils import concatenate, ufunc
from .from_set import WrappedGenerator


class Filter(WrappedGenerator):
    """
    Filters an existing sample generator

    Passes through a sample, x, if function(x) > threshold
    """

    def __init__(self, source, function, threshold=0):
        super().__init__(source)
        self.function = ufunc(function)
        self.threshold = threshold
        self.buffer = []

    def generate_samples(self, N: Union[int, float]):
        if float(N) == float(np.inf):
            samples = self.generate_source_samples(N)
            return samples[self.function(samples) > self.threshold]

        buffer_list = self.buffer.copy()
        while len(buffer_list) < N:
            samples = self.generate_source_samples(N - len(self.buffer))
            passes = self.function(samples) > self.threshold
            buffer_list = concatenate(buffer_list, samples[passes])

        self.buffer = buffer_list[N:]
        return buffer_list[:N]
