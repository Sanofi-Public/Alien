from abc import ABCMeta, abstractmethod

import numpy as np


class SampleGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate_samples(self, N):
        """
        Generates and returns N samples.

        :param N: usually an integer. Different generators
            will interpret N == inf in different ways. It
            will typically return "all" samples, perhaps as
            an iterable.
        """
        raise NotImplementedError

    def generate_sample(self):
        "Generates and returns a single sample"
        return self.generate_samples(1)[0]
