import numpy as np
from .generator import SampleGenerator
from ..data import TupleDataset, DictDataset


class UniformSampleGenerator(SampleGenerator):
    """
    Generates uniformly random samples

    :param low, high: Samples will be returned in the range(s)
        [low, high) (exclusive at the upper end). The remaining
        dimensions (of low or high) determine the shapes of the
        returned samples.
    :param dtype: the dtype of the returned values. Can be
        anything Numpy recognizes as a dtype.
    :param random_seed: a random seed to initialize the RNG.
    """

    def __init__(self, low=0, high=1, dtype=float, random_seed=None):
        self.mins = np.array(low)
        self.spans = np.array(high) - self.mins
        self.shape = self.mins.shape
        self.dtype = dtype
        self.random_seed = random_seed

    def generate_samples(self, N):
        return (
            np.random.default_rng(self.random_seed).random((N, *self.shape)) * self.spans
            + self.mins
        ).astype(self.dtype)


class RandomSampleGenerator(SampleGenerator):
    """
    :param distribution: a string indicating the type of
        distribution to sample from. Must be the name of one of
        the distributions of numpy.random.Generator. See

        https://numpy.org/doc/stable/reference/random/generator.html#distributions

    :param *args, **kwargs: arguments passed to the distribution
        during sample generation. Typically, these parametrize
        the distribution. (See the reference provided above.)

        We allow args (and kwargs) to be arrays, so that
        each corresponding term in the sample can be sampled
        with different parameters. (Note that numpy distributions
        don't normally allow this.)

    If the shape of each of the args is var_shape, then the
    shape of a single sample will be

        var_shape × shape

    with the latter dimension repeating the parameters
    given in the earlier dimensions.

    :param shape: as explained above, the shape of the
        latter dimensions of a sample, with repeated
        parameters.

    """

    def __init__(
        self, distribution="normal", *args, shape=(), dtype=float, random_seed=None, **kwargs
    ):
        rng = np.random.default_rng(random_seed)
        if not hasattr(rng, distribution):
            raise ValueError("distribution must be a valid numpy.random.Generator distribution.")
        self.dist = getattr(rng, distribution)
        self.dtype = dtype
        self.shape = shape

        # Here we determine the shape of the parameter space,
        # i.e.,
        if len(args) > 0:
            self.var_shape = self.args[0].shape
        elif len(kwargs) > 0:
            self.var_shape = self.kwargs.values()[0].shape
        else:
            self.var_shape = ()

        if len(self.var_shape) > 0:
            self.args = TupleDataset((a.flatten() for a in args))
            self.kwargs = DictDataset({k: v.flatten() for k, v in kwargs.items()})
        else:
            self.var_shape = False
            self.args = args
            self.kwargs = kwargs

    def generate_samples(self, N):
        if self.var_shape:
            # This is some array-manipulation black magic, so I should explain it.

            # First, we iterate through each arg (and kwarg) in concert,
            # yielding sets of args for the distribution. Each arg-set
            # generates a sample of shape:
            #
            #        self.shape x N
            var_samples = [
                self.dist(*args, size=(*self.shape, N), **kwargs)
                for args, kwargs in zip(iter(self.args), iter(self.kwargs))
            ]

            # We then stack these samples (one for each arg-set) on a new axis 0,
            # yielding a single large sample of shape:
            #
            #     (flattened var_shape) x self.shape x N
            samples = np.stack(var_samples)

            # We then reshape this sample, reconstituting the initial dimensions
            # into self.var_shape.
            samples = samples.reshape((*self.var_shape, *self.shape, N))

            # Reshaping is sensitive to how the array is ordered in memory.
            # That the reshape works correctly depends on
            # 1. the fact that the reshaped dimensions come at the beginning, and
            # 2. the fact that these are standard C-ordered arrays.

            # But this leaves us with the batch dimension (size N) at the end,
            # and it needs to be at the beginning. Hence,
            return np.moveaxis(samples, -1, 0).astype(self.dtype)
        else:
            return self.dist(*self.args, size=(N, *self.shape), **self.kwargs).astype(self.dtype)
