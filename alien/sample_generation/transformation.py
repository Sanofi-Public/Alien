from ..utils import ufunc
from .from_set import WrappedGenerator


class TransformedSampleGenerator(WrappedGenerator):
    """
    Wraps another sample generator. If the wrapped generator
    yields a sample x, then this generator yields sample
    function(x).

    function may be a numpy vectorized function, or any python function.
    It will be vectorized if it raises a TypeError on the first call.
    """

    def __init__(self, source, function):
        super().__init__(source)
        self.function = ufunc(function)

    def generate_samples(self, N, verbose=True):
        if verbose:
            print("Generating and transforming samples...", end="")
        samples = self.function(self.generate_source_samples(N))
        if verbose:
            print("Done")
        return samples
