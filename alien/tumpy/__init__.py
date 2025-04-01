import os
import sys
from types import SimpleNamespace

import numpy as np

from .index import names


class TorchNamespace(SimpleNamespace):
    loaded = False

    def __getattr__(self, a):
        if not TorchNamespace.loaded and a in names:
            fill_torch_namespace()
            TorchNamespace.loaded = True
        try:
            return self.__dict__[a]
        except KeyError:
            raise AttributeError(a)


def get_torch_dict():
    # path = os.path.join(os.path.dirname(__file__), 'bindings.py')
    from .torch_bindings import Tumpy

    return {n: f for n, f in Tumpy.__dict__.items() if n in names}


def fill_torch_namespace():
    torch.__dict__.update(get_torch_dict())


def fill_numpy_namespace():
    from .numpy_bindings import Numpy

    for n in names:
        if hasattr(Numpy, n):
            numpy.__dict__[n] = Numpy.__dict__[n]
        elif hasattr(np, n):
            numpy.__dict__[n] = np.__dict__.get(n)  # , print(f"Can't find np.{n}"))#Numpy.__dict__[n])


def clear_namespace():
    for n in names:
        globals().pop(n, None)
    global backend
    backend = None


def set_backend(bend):
    """
    Sets (or changes) the backend.
    `bend` can be `'torch'` or `'numpy'`.
    """
    global backend
    assert bend in {"torch", "numpy"}
    if bend == backend:
        return

    if backend in {"numpy", "torch"}:
        clear_namespace()

    if bend == "torch":
        if len(torch.__dict__) < 10:
            fill_torch_namespace()
        globals().update(torch.__dict__)

    elif bend == "numpy":
        globals().update(numpy.__dict__)

    backend = bend


torch = TorchNamespace()
numpy = SimpleNamespace()

backend = None
fill_numpy_namespace()
set_backend("torch" if "torch" in sys.modules else "numpy")
tumpy = sys.modules[__name__]
