from collections.abc import Mapping, MutableSet, MutableSequence, Collection
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike


# pylint: disable=import-error
def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, Collection) and not isinstance(x, str):
        return list(x)
    if x is None:
        return []
    return [x]


def match(target, pool, fn=lambda x, y: x == y):
    for p in pool:
        if fn(target, p):
            return p
    return None


def isint(i):
    """Check whether i can be cast to an integer

    Args:
        i (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return int(i) == i
    except (ValueError, TypeError):
        return False


def dict_get(d, *keys, _pop=False, **kwargs):
    out = {}
    d_get = d.pop if _pop else d.get
    for k in keys:
        if isinstance(k, list) or isinstance(k, tuple):
            for kk in reversed(k):
                if kk in d:
                    out[k[0]] = d_get(kk)
        elif k in d:
            out[k] = d_get(k)
    for k, v in kwargs.items():
        if k in d:
            out[k] = d_get(k) 
        elif k not in out:
            out[k] = v
    return out


def dict_pop(d, *keys, **kwargs):
    return dict_get(d, *keys, _pop=True, **kwargs)


def std_keys(d, *groups, **defaults):
    """
    Standardizes keys in a dictionary.
    Each positional arg, `g`, apart from the first one (`d`) should be an 
    iterable of keys. Then `d` is processed so that all keys in `g`
    are replaced with the first key in `g`. If more than one key in `g`
    is in `d`, then all keys in `g` except the first (in `d`) will be removed.
    """
    out = d.copy()
    for g in groups:
        for h in reversed(g):
            if h in d:
                del out[h]
                out[g[0]] = d[h]
    for k, v in defaults.items():
        if k not in out:
            out[k] = v
    return out


def update_copy(d1, d2=None, **kwargs):
    d = d1.copy()
    if isinstance(d2, dict):
        d.update(d2)
    d.update(kwargs)
    return d


ERROR = "f6b1c433450cb749f45844dd60ab3b27"

def any_get(s, elements, default=ERROR):
    assert isinstance(elements, Collection) and not isinstance(elements, str)
    assert len(elements) > 0

    if isinstance(s, Mapping):
        try:
            return s[any_get(s.keys(), elements, default=ERROR)]
        except KeyError:
            pass
    else:
        for e in elements:
            if e in s:
                return e
    
    if default==ERROR:
        raise KeyError(f'None of the given keys are in the {type(s)}.')
    else:
        return default


def any_pop(s, keys, default=ERROR):
    try:
        if isinstance(s, Mapping):
            k = any_get(s.keys(), keys, default=ERROR)
            return s.pop(k)
        else:
            k = any_get(s, keys, default=ERROR)
            s.remove(k)
            return k

    except KeyError:
        if default==ERROR:
            raise KeyError(f'None of the given keys are in the {type(s)}.')
        else:
            return default


def alias(argname):
    """Aliases an argument"""
    # TODO: write argument aliasing tool
    #raise NotImplementedError


def multisearch(a, query, one_to_one=True):
    """
    Finds the indices of multiple query values. Searches over the
    first axis of 'a', with further axes corresponding to 'feature
    space', i.e., the shape of the search terms. Return type depends
    on one_to_one.

    :param a: the array to search
    :param query: an array of query values
    :one_to_one: if True, validates that each search term
        appears exactly once, and returns a corresponding array
        of indices. If False, returns a 2D array with 2nd axis of
        length 2, with each pair of the form (query_index, array_index)
    """
    red_axes = tuple(range(-a.ndim + 1, 0))  # 'feature space' axes
    hits = np.all(a == query[:, None, ...], axis=red_axes)
    args = np.argwhere(hits)
    if one_to_one:
        assert np.all(
            args[:, 0] == np.arange(len(query))
        ), "Search results are not one-to-one with search queries!"
        return args[:, 1]
    else:
        return args


def ufunc(f):
    def wrapped_f(*args, **kwargs):
        try:
            return wrapped_f._f(*args, **kwargs)
        except TypeError:
            wrapped_f._f = np.frompyfunc(f, len(args), 1)
            return wrapped_f._f(*args, **kwargs)

    wrapped_f._f = f
    return wrapped_f


def shift_seed(seed, shift):
    try:
        return seed + shift
    except TypeError:
        return seed


def ranges(*args):
    """
    Takes arguments ([start,] stop, step).
    Returns a list of pairs consisting of
    (start_i, stop_i), which together divide
    up range(start, stop) into chunks of size
    step (plus final chunk).
    """
    if len(args) == 2:
        args = (0,) + args
    stop = args[-2]
    edges = list(range(*args)) + [stop]
    return list(zip(edges[:-1], edges[1:]))


class chunks:
    """
    Takes arguments (seq, [[start,] stop,] step).
    Returns an iterator which iterates over chunks
    of seq, of size step.
    """

    def __init__(self, seq, *args):
        if len(args) == 1:
            try:
                args = (len(seq),) + args
            except TypeError:
                seq = list(seq)
                args = (len(seq),) + args
        self.seq = seq
        self.range_iter = iter(ranges(*args))
        self.args = args
        self.step = args[-1]

    def __len__(self):
        return (len(self.seq) + self.step - 1) // self.step

    def __iter__(self):
        self.range_iter = iter(ranges(*self.args))
        return self

    def __next__(self):
        start, stop = next(self.range_iter)
        return self.seq[start:stop]


def frac_enum(seq, start_zero=True):
    """
    Much like enumerate, returns an iterator yielding ordered pairs
    (t, x) where x is an element of seq and t is the fraction of the
    way through the sequence.
    if start_zero==True, the fraction starts at 0 and ends just
    short of 1. Otherwise, starts just over 0 and ends at 1.
    """
    fracs, step = np.linspace(0, 1, len(seq), endpoint=False, retstep=True)
    if not start_zero:
        fracs += step
    return zip(fracs, seq)


def add_slice(s: slice, i: int) -> slice:
    """
    Returns a 'shifted' slice `slice(s.start + i, s.stop + i, s.step)`,
    unless `s` is `None`, in which case it returns a slice representing
    the whole window, minus a bit at the start (if `i > 0`) or the end
    (if `i < 0`).
    """
    if s is None:
        s = slice(None)

    if s.start is None:
        start = i if i > 0 else None
    else:
        start = s.start + i

    if s.stop is None:
        stop = i if i < 0 else None
    else:
        stop = s.stop + i

    return slice(start, stop, s.step)


def reshape(x, shape, index=None):
    if isinstance(index, slice):
        assert index.step == 1, f"index step must be 1, but yours is {index.step}"
        shape = x.shape[: index.start] + shape + x.shape[index.stop :]

    try:
        return x.reshape(*shape)
    except AttributeError:
        pass

    if "tensorflow" in str(type(x)):
        import tensorflow as tf

        return tf.reshape(x, shape)

    raise TypeError(f"Can't reshape tensors of type {type(x)}")


def flatten(a, dims):
    if dims < 0:
        return reshape(a, (1 - dims) * (1,) + a.shape)
    return reshape(a, (-1,) + a.shape[dims:])


def diagonal(x, dims=2, degree=1, bdim=0):
    bshape = x.shape[:bdim]
    mshape = x.shape[bdim : bdim + degree]
    mlen = np.prod(mshape)
    fshape = x.shape[bdim + dims * degree :]
    x = reshape(x, bshape + dims * (mlen,) + fshape)
    for _ in range(dims - 1):
        x = np.diagonal(x, axis1=bdim, axis2=bdim + 1)
    return x.reshape(bshape + mshape + fshape)


def concatenate(*args):
    """
    Concatenates a series of datasets, or one of the supported
    datatypes, along axis 0 (the samples axis)
    """
    args = [a for a in args if a is not None and a != []]
    if len(args) == 0:
        return None
    from .data.dataset import TeachableDataset, TeachableWrapperDataset

    if all(isinstance(a, TeachableWrapperDataset) for a in args):
        return TeachableDataset.from_data(concatenate(*(a.data for a in args)))
    elif all(isinstance(a, np.ndarray) for a in args):
        return np.concatenate(args, axis=0)
    elif all("torch" in str(type(a)) for a in args):
        import torch

        return torch.cat(args, dim=0)
    elif all(isinstance(a, tuple) for a in args):
        return tuple(concatenate(*vals) for vals in zip(*args))
    elif all(isinstance(a, dict) for a in args):
        return {k: concatenate(*(d[k] for d in args)) for k in args[0].keys()}
    elif all(isinstance(a, MutableSequence) for a in args):
        return sum(args, [])
    else:
        raise TypeError("Unsupported types for concatenate function.")


def join(*args, make_ds=False):
    """
    Concatenates a series of datasets along axis 1 (the first
    feature axis). Datasets must have same length, and if they are
    numpy or torch arrays, they must have the same shape in
    dimensions >= 2.
    """
    # pylint: disable=import-outside-toplevel
    from .data.dataset import TeachableDataset, TeachableWrapperDataset

    if len(args) == 1 and is_iterable(args[0]):
        args = args[0]

    assert all(len(a) == len(args[0]) for a in args[1:])
    args = [a.data if isinstance(a, TeachableWrapperDataset) and (make_ds := True) else a for a in args]


    if any(type(a) == tuple for a in args):
        args_unpacked = []
        for a in args:
            if isinstance(a, tuple):
                args_unpacked += [*a]
            else:
                args_unpacked += [a]
        data = tuple(args_unpacked)
    elif any(isinstance(a, np.ndarray) for a in args):
        args = [(a[...,None] if a.ndim < 2 else a) for a in args]
        data = np.concatenate(args, axis=1)
    elif all("torch" in str(type(a)) for a in args):
        import torch
        args = [(a[...,None] if a.ndim < 2 else a) for a in args]
        data = torch.cat(args, dim=1)
    elif all(isinstance(a, dict) for a in args):
        data = {}
        for a in args:
            data.update(a)

    return TeachableDataset.from_data(data) if make_ds else data


def is_iterable(x):
    try:
        iter(x)
        return True
    except RuntimeError:
        return False


def as_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    from .data import ArrayDataset

    t = str(type(data))
    if "torch" in t:
        return data.cpu().detach().numpy()
    elif "tensorflow" in t:
        return data.numpy()
    elif isinstance(data, ArrayDataset):
        return data.__array__()
    else:
        return np.asarray(data)


def is_one(x):
    return x == 1 and type(x) == int


def zip_dict(*dicts):
    """Similar behavior of zip(*) for dictionaries.
    Assumes that all dicts have the same keys.
    >>> zip_dict({'a': 1}, {'a': 2})
    {'a': (1, 2)}

    Returns:
        _type_: _description_
    """
    return {k: (d[k] for d in dicts) for k in dicts[0].keys()}


def axes_except(X, non_axes):
    if isint(non_axes):
        non_axes = (non_axes,)
    non_axes = tuple(a % X.ndim for a in non_axes)
    return tuple(a for a in range(X.ndim) if a not in non_axes)


def sum_except(X, non_axes):
    return X.sum(axis=axes_except(X, non_axes))


def no_default():
    raise NotImplementedError("no_default is used as a unique reference, and should not be called")


class SelfDict(dict):
    """
    Subclass of dict class which allows you to refer to,
    eg., d['attr'] as d.attr and vice versa.

    You can also index with lists, where
    d[['a', 'b']] == {'a': d['a'], 'b': d['b']}
    Similarly with pop()
    """

    def __init__(self, *args, default=no_default, **kwargs):
        super().__init__(*args)
        self.update(kwargs)
        self.__dict__ = self
        for k, v in self.items():
            if isinstance(v, Mapping):
                self[k] = SelfDict(v)

    def __setitem__(self, key, value):
        if isinstance(value, Mapping):
            super().__setitem__(key, SelfDict(value))
        else:
            super().__setitem__(key, value)

    def __setattr__(self, key, value):
        if key.startswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __getitem__(self, key):
        s = super()
        if isinstance(key, list):
            # if self.__default == no_default:
            return SelfDict({k: s.__getitem__(k) for k in key if k in s.__iter__()})
            # else:
            #    return SelfDict({k:(s[k] if k in s else self.__default) for k in key})
        else:
            # if key not in self and self.__default != no_default:
            #    return self.__default
            return s.__getitem__(key)

    def pop(self, key, default=no_default):
        s = super()
        # default == self.__default if default == no_default else default
        if isinstance(key, list):
            if default == no_default:
                return SelfDict({k: s.pop(k) for k in key if k in s.__iter__()})
            else:
                return SelfDict({k: s.pop(k, default) for k in key})
        else:
            return s.pop(key) if default == no_default else s.pop(key, default)


class CachedDict:

    def __init__(self, get, *init_keys, **d2):
        self.cache = {}
        self._get = get.__get__(self)

        if isinstance(init_keys[0], Mapping):
            d1, init_keys = init_keys[0], init_keys[1:]
        else:
            d1 = {}
        
        self.cache.update(d1)
        self.cache.update(d2)

        if len(init_keys) == 1 and (
                isinstance(init_keys[0], MutableSet) or 
                isinstance(init_keys[0], MutableSequence)):
            init_keys = init_keys[0]

        [self(k) for k in init_keys]

        for m in ['keys', 'values', 'items', 'get', 'pop', '__iter__', '__contains__']:
            self.__dict__[m] = getattr(self.cache, m)

    def __call__(self, key):
        if key in self.cache:
            return self.cache[key]

        value = self._get(key)
        self.cache[key] = value
        return value

    __getitem__ = __getattr__ = __call__


def dot_last(a, b):
    return (a * b).sum(axis=-1)
