import numpy as np

from ..data import Dataset
from ..utils import concatenate
from .generator import SampleGenerator


class SetSampleGenerator(SampleGenerator):
    """
    Takes samples from a given dataset. Can remove samples upon request
    (eg., if they are ultimately used).

    :param data: dataset which implements __getitem__ (with advanced
        slicing), __len__, and find
    :param Xy: True if dataset contains both X and y values. Samples
        will return from self.data[:,:-1]
    """

    def __init__(
        self,
        data,
        shuffle: bool = True,
        random_seed=None,
        cap_to_size: bool = True,
        Xy: bool = False,
    ):
        super().__init__()
        if not isinstance(data, Dataset):
            data = Dataset.from_data(data)

        self.Xy = Xy
        self.data = data
        self.cap_to_size = cap_to_size

        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.random_seed = random_seed
        self.reshuffle()

    def reshuffle(self):
        """Reshuffles current indices"""
        if self.shuffle:
            np.random.default_rng(self.random_seed).shuffle(self.indices)
        self.pointer = 0

    @property
    def labels(self):
        return self.data.y

    def generate_samples(self, N=np.inf, reshuffle=False):
        if reshuffle:
            self.reshuffle()

        length = len(self.indices)

        if np.isinf(N) or self.cap_to_size and N > length:
            N = length

        if N + self.pointer > length:
            n_1 = length - self.pointer
            n_2 = N - n_1
            return concatenate(self.generate_samples(n_1), self.generate_samples(n_2))

        samples = self.data[self.indices[self.pointer : self.pointer + N]]

        self.pointer += N
        if self.pointer == length:
            self.reshuffle()
        return samples

    def remove_samples(self, samples):
        """
        'Removes' or, rather, *hides* samples from this generator.
        Hidden samples are still stored in self.data, but will not
        appear in any future calls to generate_samples.
        """
        data_indices = []
        for s in samples:
            data_indices.extend(list(self.data.find(s, first=False)))

        self.remove_data_indices(data_indices)

    def remove_sample(self, sample):
        "Single-sample version of remove_samples"
        self.remove_samples([sample])

    def remove_data_indices(self, indices):
        """Remove data indices and shift self.pointer accordingly"""
        shuffle_indices = []
        for d_i in indices:
            s_i = np.argwhere(self.indices == d_i)
            if len(s_i):
                shuffle_indices.append(s_i[0, 0])
        shuffle_indices = list(set(shuffle_indices))
        shuffle_indices = np.array(shuffle_indices, dtype=int)

        self.pointer -= np.sum(shuffle_indices < self.pointer)
        self.indices = np.delete(self.indices, shuffle_indices)

    def __len__(self):
        return len(self.indices)


class WrappedGenerator(SampleGenerator):
    """Generator wrapper class."""

    def __init__(self, source, random_seed=None):
        if not hasattr(source, "generate_samples"):
            source = SetSampleGenerator(source, random_seed=random_seed)
        self.source = source
        self.generate_source_samples = self.source.generate_samples
        self.random_seed = random_seed

    def generate_samples(self, N):
        raise NotImplementedError
