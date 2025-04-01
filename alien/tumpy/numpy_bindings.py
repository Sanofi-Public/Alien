import sys

import numpy as np

None_slice = slice(None)


# These are 'extra' functions we include with numpy:
class Numpy:
    def set_default_device(device):
        if device not in {"cpu", None}:
            raise ValueError(
                "Tumpy is using actual Numpy right now, so can only compute on device='cpu' "
                " or device=None. "
                "If you want to use a GPU, first call tumpy.set_backend('torch')"
            )
        Numpy.default_device = device

    default_device = "cpu"

    @staticmethod
    def get_default_device():
        return Numpy.default_device

    def to(a, d):
        return np.asarray(a)

    def device(x):
        return "cpu"

    @staticmethod
    def no_grad():
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def softmax(x, axis=None):
        from scipy.special import softmax

        return softmax(x, axis=axis)

    def is_float(x):
        if isinstance(x, np.ndarray):
            return x.dtype.kind not in "iu"
        
        if "torch" in sys.modules:
            import torch
            if isinstance(x, torch.Tensor):
                return torch.is_floating_point(x) or torch.is_complex(x)
            
        if "tensorflow" in sys.modules:
            import tensorflow as tf
            if isinstance(x, tf.Tensor):
                return x.dtype.is_floating or x.dtype.is_complex

    def is_integer(x):
        return not Numpy.is_float(x)
    
    def is_bool(x):
        if x.dtype == bool:
            return True
        
        if "torch" in sys.modules:
            import torch
            if isinstance(x, torch.Tensor):
                return x.dtype == torch.bool
    
    def numpy_dtype(x):
        if x == bool:
            return bool
        elif isinstance(x, np.ndarray):
            return x.dtype
        elif "torch" in sys.modules:
            from .torch_bindings import dtype_t_to_n
            try:
                return dtype_t_to_n[x.dtype]
            except KeyError:
                pass
        elif "tensorflow" in sys.modules:
            return x.dtype.as_numpy_dtype
        
        raise TypeError(f"Can't extract dtype from a {type(x)}")

    def is_array(x):
        if isinstance(x, np.ndarray):
            return True
        elif "torch" in sys.modules:
            import torch

            return isinstance(x, torch.Tensor)
        return False

    # Array creation functions should take a (dummy) `device` argument
    def arange(*args, device=None, **kwargs):
        return np.arange(*args, **kwargs)

    def array(*args, device=None, **kwargs):
        return np.array(*args, **kwargs)

    def asarray(*args, device=None, **kwargs):
        return np.asarray(*args, **kwargs)

    def zeros(*args, device=None, **kwargs):
        return np.zeros(*args, **kwargs)

    def empty(*args, device=None, **kwargs):
        return np.empty(*args, **kwargs)

    def fromiter(*args, device=None, **kwargs):
        return np.fromiter(*args, **kwargs)

    def ones(*args, device=None, **kwargs):
        return np.ones(*args, **kwargs)

    def full(*args, device=None, **kwargs):
        return np.full(*args, **kwargs)

    def linspace(*args, device=None, **kwargs):
        return np.linspace(*args, **kwargs)

    def narrow(arr, axis, start, length):
        """
        Simulates torch.narrow functionality in NumPy.

        Args:
        - arr: Input NumPy array.
        - dimension: Dimension along which to narrow the array.
        - start: Starting index for narrowing.
        - length: Length of the narrow dimension.

        Returns:
        - Narrowed array.
        """
        if axis < 0:
            axis += arr.ndim
        indices = (None_slice,) * axis + (slice(start, start + length),)
        return arr[tuple(indices)]
    
    