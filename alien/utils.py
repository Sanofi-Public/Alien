import os
import sys
from collections.abc import Collection, Mapping, MutableSequence, MutableSet
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .tumpy import tumpy as tp


# pylint: disable=import-error,import-outside-toplevel
def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)

    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)


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


def isint(i) -> bool:
    """Check whether i can be cast to an integer

    Args:
        i (_type_): _description_

    Returns:
        bool: _description_
    """
    try:
        return int(i) == i
    except (ValueError, TypeError):
        return False


def is_0d(i):
    """
    Checks whether i is 0-dimensional, i.e., if it's a number, returns True;
    If it's a 0d array, returns True. If it's a higher-D array, returns False
    """
    if isinstance(i, (slice, list)):
        return False
    if isinstance(i, tuple):
        return all(is_0d(j) for j in i)
    try:
        return i.ndim == 0
    except AttributeError:
        return True


def dict_get(input_dict, *keys, _pop=False, **kwargs):
    d_get = input_dict.pop if _pop else input_dict.get
    out = _dict_get_out(input_dict, *keys, _pop=_pop)
    for k, v in kwargs.items():
        if k in input_dict:
            out[k] = d_get(k)
        elif k not in out:
            out[k] = v
    return out


def _dict_get_out(input_dict, *keys, _pop=False):
    d_get = input_dict.pop if _pop else input_dict.get
    out = {}
    for k in keys:
        if isinstance(k, list) or isinstance(k, tuple):
            for inner_key in reversed(k):
                if inner_key in input_dict:
                    out[k[0]] = d_get(inner_key)
        elif k in input_dict:
            out[k] = d_get(k)
    return out


def dict_pop(input_dict, *keys, **kwargs):
    return dict_get(input_dict, *keys, _pop=True, **kwargs)


def std_keys(input_dict, *groups, **defaults):
    """
    Standardizes keys in a dictionary.
    Each positional arg, `g`, apart from the first one (`d`) should be an
    iterable of keys. Then `d` is processed so that all keys in `g`
    are replaced with the first key in `g`. If more than one key in `g`
    is in `d`, then all keys in `g` except the first (in `d`) will be removed.
    """
    out = input_dict.copy()
    for g in groups:
        for h in reversed(g):
            if h in input_dict:
                del out[h]
                out[g[0]] = input_dict[h]
    for k, v in defaults.items():
        if k not in out:
            out[k] = v
    return out


class Peekable:
    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self._next = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            return next(self.iterator)
        out = self._next
        self._next = None
        return out

    def peek(self):
        if self._next is None:
            self._next = next(self.iterator)
        return self._next


def update_copy(d_1, d_2=None, **kwargs):
    d = d_1.copy()
    if isinstance(d_2, dict):
        d.update(d_2)
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

    if default == ERROR:
        raise KeyError(f"None of the given keys are in the {type(s)}.")
    return default


def any_pop(s, keys, default=ERROR):
    try:
        if isinstance(s, Mapping):
            k = any_get(s.keys(), keys, default=ERROR)
            return s.pop(k)
        k = any_get(s, keys, default=ERROR)
        s.remove(k)
        return k

    except KeyError:
        if default == ERROR:
            raise KeyError(f"None of the given keys are in the {type(s)}.")
        return default


def alias(argname):
    """Aliases an argument"""
    # TODO: write argument aliasing tool
    raise NotImplementedError


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
    hits = tp.all(a == query[:, None, ...], axis=red_axes)
    args = tp.argwhere(hits)
    if one_to_one:
        assert tp.all(args[:, 0] == tp.arange(len(query))), "Search results are not one-to-one with search queries!"
        return args[:, 1]
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


None_slice = slice(None)


def default_min(*args, default=None):
    args = set(args)
    args.discard(default)
    if len(args) == 0:
        return default
    return min(args)


def default_max(*args, default=None):
    args = set(args)
    args.discard(default)
    if len(args) == 0:
        return default
    return max(args)


def expand_ellipsis(index, ndim):
    is_ell = tuple(... is x for x in index)
    if any(is_ell):
        if isinstance(ndim, tuple):
            ndim = len(ndim)
        i_e = is_ell.index(True)
        index = index[:i_e] + (None_slice,) * (ndim + 1 - len(index)) + index[i_e + 1 :]
    return index


def new_shape(old_shape, indices):
    """
    If A.shape = old_shape, then new_shape returns A[indices].shape.
    No big arrays like A or A[indices] are actually created.
    """
    from itertools import zip_longest

    indices = expand_ellipsis(indices, old_shape)

    shape = []
    for d, i in zip_longest(old_shape, indices, fillvalue=None_slice):
        if i == None_slice:
            shape.append(d)

        elif isinstance(i, slice):
            start = i.start or 0
            stop = min(1e15 if i.stop is None else i.stop, d)
            step = i.step or 1
            shape.append(int((stop - 1 - start) // step + 1))

        elif tp.is_array(i) or isinstance(i, list):
            shape.append(len(i))

        else:
            pass  # this dimension will disappear

    return tuple(shape)


def move_index(index, axis0, axis1):
    if tp.isin(..., index):
        i_e = tp.argwhere(index == ...)[0].item()
        indices = indices[:i_e] + (None_slice,) * (self.ndim + 1 - len(indices)) + indices[i_e + 1 :]
    if axis0 < 0:
        axis0 += len(index)
    if axis1 < 0:
        axis1 += max(axis0 + 1, len(index))
    if (m := max(axis0, axis1)) >= len(index):
        index = index + (None_slice,) * (m + 1 - len(index))
    i_0 = index[axis0]
    index = index[:axis0] + index[axis0 + 1 :]
    index = index[:axis1] + (i_0,) + index[axis1:]
    return index


def shift_seed(seed, shift):
    try:
        if not seed:
            return shift
        return seed + shift
    except TypeError:
        return seed


def version_number(x):
    x = getattr(x, "__version__", x)
    if isinstance(x, str):
        v = []
        for n in x.split("."):
            try:
                v.append(int(n))
            except Exception as _:
                v.append(0.5)
        return tuple(v)
    return x


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
    Takes arguments `(seq, [[start,] stop,] step)`.
    Returns an iterator which iterates over chunks
    of `seq`, of size `step`.

    If the first argument is an integer, it assumes
    no sequence was provided, so that the effective
    signature is `([[start,] stop], step])` and uses
    `range(stop)` instead.
    """

    def __init__(self, seq, *args):
        if isinstance(seq, int):
            args = (seq,) + args
            seq = list(range(args[-2]))
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
    fracs, step = tp.linspace(0, 1, len(seq), endpoint=False, retstep=True)
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
    except TypeError:
        return x.reshape(shape)

    if "tensorflow" in str(type(x)):
        import tensorflow as tf

        return tf.reshape(x, shape)

    raise TypeError(f"Can't reshape tensors of type {type(x)}")


def flatten(a, dims):
    if dims < 0:
        return reshape(a, (1 - dims) * (1,) + a.shape)
    return reshape(a, (-1,) + a.shape[dims:])


def diagonal(M, axes=(-2, -1)):
    """
    Returns a writeable view into the diagonal of `M`.
    The diagonal will be taken along axes `axes` (default
    `(-2,-1)`), and appended to the end of the shape, while the other
    axes are left unchanged.

    Doesn't check if all the given axes are the same size. If not,
    behaviour is unpredictable and probably bad!
    """
    # Pytorch gives a writeable view already
    if 'torch' in str(type(M)):
        import torch
        return torch.diagonal(M, dim1=axes[0], dim2=axes[1])

    M = np.asarray(M)
    axes = tuple(i % M.ndim for i in axes)
    strides = tuple(s for i, s in enumerate(M.strides) if i not in axes) + (sum(M.strides[i] for i in axes),)

    shape = tuple(d for i, d in enumerate(M.strides) if i not in axes) + (M.shape[axes[0]],)

    return as_strided(M, strides=strides, shape=shape)


def view(x, shape):
    """Return a view of `x` with a new shape. Will never
    copy data, so writing to the view will write to `x`."""
    if isinstance(shape, np.ndarray):
        y = x[:]
        y.shape = shape
        return y
    else:  # Assume torch tensor:
        return x.view(*shape)


def concatenate(*args):
    """
    Concatenates a series of datasets, or one of the supported
    datatypes, along axis 0 (the samples axis)
    """
    args = [a for a in args if a is not None and len(a)]
    if len(args) == 0:
        return None
    from .data.dataset import TeachableDataset, TeachableWrapperDataset

    if all(isinstance(a, TeachableWrapperDataset) for a in args):
        return TeachableDataset.from_data(concatenate(*(a.data for a in args)))
    elif all(isinstance(a, np.ndarray) for a in args):
        return np.concatenate(args, axis=0)
    elif all("torch" in str(type(a)) for a in args):
        # pylint: disable=import-outside-toplevel
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
    from .data.dataset import TeachableDataset

    args = _join_helper_args(args)

    if any(type(a) == tuple for a in args):
        data = _join_helper_tuple(args)
    elif any(isinstance(a, np.ndarray) for a in args):
        data = _join_helper_numpy(args)
    elif any("torch" in str(type(a)) for a in args):
        data = _join_helper_torch(args)
    elif any("Tumpy" in str(type(a)) for a in args):
        data = _join_helper_tumpy(args)
    elif all(isinstance(a, dict) for a in args):
        data = {}
        for a in args:
            data.update(a)
    elif all(isinstance(a, list) for a in args):
        return list(list(c) for c in zip(args))

    return TeachableDataset.from_data(data)


def _join_helper_args(args):
    from .data.dataset import TeachableWrapperDataset

    if len(args) == 1 and is_iterable(args[0]):
        args = args[0]

    assert all(len(a) == len(args[0]) for a in args[1:])
    args = [a.data if isinstance(a, TeachableWrapperDataset) else a for a in args]
    return args


def _join_helper_tuple(args):
    args_unpacked = []
    for a in args:
        if isinstance(a, tuple):
            args_unpacked += [*a]
        else:
            args_unpacked += [a]
    data = tuple(args_unpacked)
    return data


def _join_helper_numpy(args):
    if any(a.dtype == object for a in args):
        data = np.array(
            [np.concatenate([arg[i] for arg in args], axis=-1) for i in range(len(args[0]))],
            dtype=object,
        )
    else:
        args = [(a[..., None] if a.ndim < 2 else a) for a in args]
        data = np.concatenate(args, axis=1)
    return data


def _join_helper_torch(args):
    import torch

    args = [(a[..., None] if a.ndim < 2 else a) for a in args]
    data = torch.cat(args, dim=1)
    return data


def _join_helper_tumpy(args):
    args = [(a[..., None] if a.ndim < 2 else a) for a in args]
    data = tp.concatenate(args, axis=1)
    return data


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


class Identity:
    """
    Instantiate this class with an argument:
    >>> Identity(obj) = obj
    You just get the argument back. Magic!
    Other arguments do nothing.
    """

    def __new__(cls, obj, *args, **kwargs):
        return obj


def no_default():
    raise NotImplementedError("no_default is used as a unique reference, and should not be called")


def softmax(x, axis=-1, min_axes=2):
    """
    Returns the softmax of `x` along `axis`.
    Unsqueezes enough dimensions to make `x.ndim >= min_axes`.
    If `x.shape[axis] == 1`, returns the sigmoid of `x`, rather than the softmax.
    """
    while axis >= x.ndim:
        x = x[..., None]
    if x.shape[axis] > 1:
        ex = tp.exp(x)
        return ex / tp.sum(ex, axis=axis, keepdims=True)
    else:
        return 1. / (1. + tp.exp(-x))


def convert_output_type(x, type0, type1, n_classes=None, smoothing=0.0):
    from .models import Output

    if type0 == type1:
        return x

    elif type0 == Output.CLASS:
        return _convert_output_class(x, type1, n_classes, smoothing)
    elif type0 == Output.LOGIT:
        return _convert_output_logit(x, type1, n_classes, smoothing)
    elif type0 == Output.PROB:
        return _convert_output_prob(x, type1, n_classes, smoothing)


def _convert_output_class(x, output_type, n_classes=None, smoothing=0.0):
    from .models import Output

    n_classes = n_classes or x.max() + 1
    onehot = x[..., None] == tp.arange(n_classes, device=tp.device(x))
    if not smoothing:
        if output_type == Output.PROB:
            return onehot.astype(tp.float32)
        else:  # type1 == Output.LOGIT:
            xx = tp.full(x.shape + (n_classes,), -tp.inf, device=tp.device(x))
            xx[onehot] = 0.0
            return xx
    else:
        zero_prob = smoothing / (n_classes - 1)
        prob = tp.full(x.shape + (n_classes,), zero_prob, device=tp.device(x))
        prob[onehot] = 1 - smoothing
        if output_type == Output.PROB:
            return prob
        else:
            return tp.log(prob)


def _convert_output_logit(x, output_type, n_classes=None, smoothing=0.0):
    from .models import Output

    if output_type == Output.CLASS:
        return x.argmax(-1)
    elif output_type == Output.PROB:
        return softmax(x, -1)


def _convert_output_prob(x, output_type, n_classes=None, smoothing=0.0):
    from .models import Output

    if output_type == Output.CLASS:
        return x.argmax(-1)
    elif output_type == Output.LOGIT:
        return tp.log(x)


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

        if len(init_keys) == 1 and (isinstance(init_keys[0], MutableSet) or isinstance(init_keys[0], MutableSequence)):
            init_keys = init_keys[0]

        [self(k) for k in init_keys]

        for m in ["keys", "values", "items", "get", "pop", "__iter__", "__contains__"]:
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


def create_directory(input_path, exist_ok=False):
    # Check if the input is a string
    if not isinstance(input_path, str):
        raise ValueError("Input path must be a string.")

    # Remove leading and trailing whitespaces
    input_path = input_path.strip()

    # Check if the input is not empty
    if not input_path:
        raise ValueError("Input path cannot be empty.")

    # Create a Path object
    sanitized_path = Path(input_path).resolve()

    # Check if the directory already exists
    if sanitized_path.exists():
        if sanitized_path.is_dir() and exist_ok:
            return
        raise FileExistsError(f"Directory '{sanitized_path}' already exists.")

    # Create the directory
    sanitized_path.mkdir(parents=True, exist_ok=exist_ok)


def list_directory(input_path):
    # Check if the input is a string
    if not isinstance(input_path, str):
        raise ValueError("Input path must be a string.")

    # Remove leading and trailing whitespaces
    input_path = input_path.strip()

    # Check if the input is not empty
    if not input_path:
        raise ValueError("Input path cannot be empty.")

    # Create a Path object from the input path
    sanitized_path = Path(input_path).resolve()

    # Check if the path exists
    if not sanitized_path.exists():
        raise FileNotFoundError(f"The path '{sanitized_path}' does not exist.")

    # Check if the path is a directory
    if not sanitized_path.is_dir():
        raise NotADirectoryError(f"The path '{sanitized_path}' is not a directory.")

    # Return a list of files in the directory
    return [entry.name for entry in sanitized_path.iterdir()]
