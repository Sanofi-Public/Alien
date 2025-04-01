"""
Module with dataset (sub-)classes for storing data.
"""

# TODO in this module:
# - join function for DictDataset
# - check join implementation for TupleDataset
# - concatenate function for TupleDataset
# - numpy warning in concatenate
# - specify exceptions
# - Dataset.from_data parameters align with TeachableDataset.from_data
# - other smaller todos throughout
import sys
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, MutableSequence
from copy import copy, deepcopy
from typing import Any, Optional, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike

from ..tumpy import tumpy as tp
from ..utils import Identity, add_slice, is_0d, reshape, update_copy

if "torch" in sys.modules:
    import torch


class Dataset(metaclass=ABCMeta):
    """
    Abstract interface to a readable dataset.
    """

    def __new__(cls, *args, **kwargs):
        if cls == Dataset:
            if "data" in kwargs:
                data = kwargs.pop("data")
            else:
                data = args[0]
            new_cls = find_dataset_class(data, *args, **kwargs)
            return new_cls.__new__(new_cls, data, *args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, *, has_Xy=None, bdim=1, **kwargs):
        super().__init__(**kwargs)
        self.has_Xy = has_Xy
        self.bdim = bdim

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def find(self, value, first=True):
        """
        Finds instances of `value` in this dataset.
        If first is True, returns the index of the first
        occurence (or None if not found), otherwise returns
        an iterable of indices of all occurences.
        """

    @property
    @abstractmethod
    def dtype(self):
        """Get the `dtype` of the contained data."""

    @abstractmethod
    def astype(self, dtype, copy=False):
        """
        Change the `dtype` of the contained data. Uses `numpy.ndarray.astype(...)`
        or `torch.Tensor.type(...)`. Note that the default is not to copy the data
        (if possible), unlike Numpy but like Pytorch.

        Args:
            dtype: The `dtype` to change to.
            copy: If `True`, always copies the data, and returns a new `Dataset`
                containing this data. If `False`, always returns the same `Dataset`
                object, whose data is converted to `dtype`
        """

    def type(self, dtype=None, copy=False):
        """
        Behaves exactly like `.astype(...)`, except that when `dtype` is not provided,
        return the `dtype` of the contained data.
        """
        if dtype is not None:
            return self.astype(dtype, copy=copy)
        else:
            return self.dtype

    def __iter__(self):
        "Default iterator implementation"
        return iter(self[i] for i in range(len(self)))

    @staticmethod
    def from_data(*args, **kwargs):
        """
        Returns a Dataset built from the given data and other args.
        Arguments and functionality are exactly like
        TeachableDataset.from_data
        In fact, at present, this method just calls
        TeachableDataset.from_data
        """
        # TODO: pylint doesn't like that the parent class uses *args, **kwargs.
        # Need to figure out a general way that doesn't break this.
        dataset = TeachableDataset.from_data(*args, **kwargs)
        return dataset

    @property
    @abstractmethod
    def shape(self):
        """Abstract method for returning shape."""

    @property
    def ndim(self):
        """Returns: int: number of dimensions"""
        return len(self.shape)

    @property
    def batch_shape(self):
        return self.shape[: self.bdim]

    @property
    def feature_shape(self):
        return self.shape[self.bdim :]

    def reshape(self, *shape, index=None, bdim=None):
        raise NotImplementedError

    def reshape_features(self, *shape, index=None):
        return self.reshape(*shape, index=add_slice(index, self.bdim), bdim=self.bdim)

    def reshape_batch(self, *shape, index=None):
        if index is None:
            index = slice(0, self.bdim)
            bdim = len(shape)
        elif isinstance(index, slice):
            assert index.step is None or index.step == 1
            index = slice(index.start, min(index.stop, self.bdim))
            bdim = self.bdim + len(shape) - (index.stop - index.start)

        return self.reshape(*shape, index=index, bdim=bdim)

    def unsqueeze(self, index):
        return self.reshape((1,), index=slice(index, index))

    def flatten_batch(self):
        return self.reshape_batch(self, np.prod(self.batch_shape))

    def swapaxes(self, *args, **kwargs):
        return TeachableDataset.from_data(self.data.swapaxes(*args, **kwargs))


def find_dataset_class(data, *args, convert_sequences=True, **kwargs):
    if data is None or isinstance(data, dict):
        return DictDataset
    if convert_sequences and isinstance(data, MutableSequence):
        data = np.asarray(data)
    return find_imported_dataset_class(data)


def find_imported_dataset_class(data):
    if isinstance(data, TeachableDataset):
        return Identity
    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            return ObjectDataset
        else:
            return NumpyDataset
    elif isinstance(data, tuple):
        return TupleDataset
    elif "torch" in str(type(data)):
        return TorchDataset
    elif "deepchem" in str(type(data)):
        from .deepchem import DeepChemDataset

        return DeepChemDataset
    elif "pandas" in str(type(data)):
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return DataFrameDataset
        elif isinstance(data, pd.Series):
            return PandasSeriesDataset
    else:
        warnings.warn("Passing an unknown data format into TeachableDataset.")
        return TeachableWrapperDataset


class TeachableDataset(Dataset):
    """
    Abstract interface to a teachable dataset.
    """

    @abstractmethod
    def append(self, x: Any):
        """
        Appends a single sample to the end of the dataset.
        """

    def extend(self, X: ArrayLike):
        """
        Appends a batch of samples to the end of the dataset.
        """
        # This is the default implementation of extend.
        # Subclasses may accomplish this faster
        for val in X:
            self.append(val)

    @staticmethod
    def from_data(
        data=None,
        shuffle: Optional[Union[bool, str]] = False,
        random_seed: Optional[Union[int, ArrayLike, SeedSequence, BitGenerator, Generator]] = None,
        recursive: bool = True,
        convert_sequences: bool = True,
        **kwargs,
    ):
        """
        Creates a TeachableDataset with given data.

        :param data: the initial data of the dataset
            Can be:
                * another TeachableDataset
                * a Python mutable sequence (eg., a list) or
                  anything that implements the interface
                * a Numpy array
                * a Pytorch tensor
                * a dictionary or tuple whose values are one of the above types
                * a Pandas DataFrame

        :param shuffle: if this evaluates to True, data will be wrapped in a shuffle,
                exposing the ShuffledDataset interface.
                Can be:
                * anything evaluating to False
                * 'identity' (initial shuffle is the identity)
                * 'random' (initial shuffle is random)

        :param random_seed: a random seed to pass to Numpy's shuffle algorithm.
                    If None (the default), Numpy gets entropy from the OS.

        :param recursive: if True, data like MutableSequences or TeachableDatasets that
                    already expose the needed interface, will still be wrapped;
                    if False, such data will be returned as-is, with no new object
                    created.
        """
        if shuffle:
            return ShuffledDataset(
                TeachableDataset.from_data(data, recursive=False, convert_sequences=convert_sequences, **kwargs),
                shuffle=shuffle,
                random_seed=random_seed,
            )
        else:
            return find_dataset_class(data, **kwargs)(data, **kwargs)

    @staticmethod
    def from_deepchem(data):
        try:
            # pylint: disable=import-outside-toplevel
            import deepchem

            assert isinstance(data, deepchem.data.Dataset)
            from .deepchem import DeepChemDataset

            return DeepChemDataset(data)
        except Exception as exc:
            raise NotImplementedError("We thought this was a DeepChem dataset, but apparently not!") from exc

    def get_shuffle(self, shuffle="random", random_seed=None):
        """Return a shuffled version of self

        Args:
            shuffle (str, optional): The initial shuffle - `'identity'` or `'random'`. Defaults to `'random`'.
            random_seed (int, optional): A random seed for the shuffle. Defaults to None.

        Returns:
            ShuffledDataset: A shuffled version of `self`
        """
        return ShuffledDataset(self, shuffle=shuffle, random_seed=random_seed)

    def swapaxes(self, *args, **kwargs):
        return TeachableDataset.from_data(self.data.swapaxes(*args, **kwargs))


class TeachableWrapperDataset(TeachableDataset):
    """
    Wraps another dataset-like object.
    Functions as an abstract base class for wrapping specific data types.
    Also functions concretely as the default wrapper for MutableSequences,
    other TeachableDatasets, and anything else which exposes a suitable
    interface.
    """

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def append(self, x):
        val = self.data.append(x)
        if val is not None:
            self.data = val

    def extend(self, X):
        try:
            val = self.data.extend(X)
            if val is not None:
                self.data = val
        except AttributeError:
            super().extend(X)

    def find(self, value: Any, first: bool = True):
        # Raising NotImplementedError to avoid missing abstract method.
        raise NotImplementedError

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            i0, *i = index
            if i0 == ...:
                try:
                    # Assume we can push '...' onto the rows
                    return [row[(..., *i)] for row in self.data]
                except LookupError:
                    # Apparently not
                    return [row[i] for row in self.data]
            try:
                # Assume i0 is an integer, so self[i0] will be
                # a single row
                return self[int(i0)][i]
            except (ValueError, TypeError):
                # Apparently not
                return [row[i] for row in self[i0]]

        try:
            return self.data[index]
        except LookupError:
            return [self.data[i] for i in index]

    def _ignore__iter__(self):
        try:
            return iter(self.data)
        except TypeError:
            return super().__iter__()

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype):
        raise NotImplementedError


class ShuffledDataset(TeachableWrapperDataset):
    """
    Presents a shuffle of an existing dataset (or MutableSequence)

    Added data goes at the end and isn't shuffled (until reshuffle() is called).

    :param data: the existing dataset to wrap
    :param shuffle: determines the initial shuffle state: 'random' or 'identity', or
        an iterable of indices.
    :param random_seed: random seed to pass to the numpy shuffle algorithm.
                  If None, get a source of randomness from the OS.
    """

    def __init__(
        self,
        data,
        shuffle="random",
        random_seed: Optional[Union[int, ArrayLike, SeedSequence, BitGenerator, Generator]] = None,
        recursive=False,
        bdim=1,
    ):
        self.rng = tp.random.default_rng(random_seed)
        assert bdim == 1, "ShuffledDataset is only possible with one batch dimension."
        super().__init__(data)
        if (not recursive) and isinstance(data, ShuffledDataset):
            self.data = data[data.shuffle]
        if isinstance(shuffle, np.ndarray):
            assert len(shuffle) == len(self.data), "Supplied shuffle must be same length as data!"
            self.shuffle = shuffle
        elif shuffle == "identity" or not shuffle:
            self.shuffle = np.arange(len(self.data))
        else:  # shuffle == 'random' OR any True-valued
            self.shuffle = np.arange(len(self.data))
            self.reshuffle()

    def reshuffle(self):
        """Reshuffles self with rng"""
        # TODO: random_seed is not used here. Should remove or refactor to use it
        self.rng.shuffle(self.shuffle)

    def extend_shuffle(self):
        """Extend self.shuffle with [len(self.shuffle), ..., len(self.data)]."""
        len_shuffle, len_data = len(self.shuffle), len(self.data)
        if len_shuffle < len_data:
            self.shuffle = np.append(self.shuffle, np.arange(len_shuffle, len_data))

    def __getitem__(self, index):
        self.extend_shuffle()

        if isinstance(index, tuple):
            i0, *i = index
            return self.data[(self.shuffle[i0], *i)]

        return self.data[self.shuffle[index]]

    def find(self, value: Any, first: bool = True):
        """Return index(es) of value in self.

        Args:
            value (Any): value to look for
            first (bool, optional): whether to return first instance of value or all of them. Defaults to True.

        Returns:
            _type_: _description_
        """
        i = self.data.find(value, first)
        if first:
            return i if i is None else self.shuffle[i]
        else:
            return i if len(i) == 0 else self.shuffle[i]

    def __iter__(self):
        self.extend_shuffle()
        return iter(self.data[self.shuffle])
        # return iter(TeachableDataset.from_data(self.data[self.shuffle]))

    def astype(self, dtype, copy=False):
        if copy:
            new = self.__class__(self.data, shuffle=self.shuffle, bdim=self.bdim)
            new.rng = deepcopy(self.rng)
            return new
        else:
            self.data = self.data.astype(dtype=dtype, copy=False)
            return self

    def __array__(self, dtype=None):
        "Converts to a Numpy array"
        return np.asarray(self.data, dtype=dtype)[self.shuffle]

    @property
    def X(self):
        try:
            X = self.data.X
        except AttributeError:
            warnings.warn(
                f"Dataset of type {type(self.data)} does not have an X attribute. Returning whole dataset instead"
            )
            X = self.data
        X = ShuffledDataset(X, shuffle=self.shuffle)
        X.rng = None
        return X

    @property
    def y(self):
        try:
            y = self.data.y
        except AttributeError as err:
            raise NotImplementedError(f"Dataset of type {type(self.data)} does not have a y attribute.") from err
        y = ShuffledDataset(self.data.y, shuffle=self.shuffle)
        y.rng = None
        return y


def compute_bdim(old_shape, old_bdim, new_shape):
    b_size = np.prod(old_shape[:old_bdim])
    size = 1
    for bdim, d in enumerate(new_shape):
        size *= d
        if size == b_size:
            return bdim + (size == b_size)
        elif size > b_size:
            raise ValueError("New shape must have initial axes with total size equal to the original batch size.")


class ArrayDataset(TeachableWrapperDataset):
    """
    Abstract base class for datasets based on numpy, pytorch,
    or other similarly-interfaced arrays.
    """

    def __getitem__(self, index):
        bdim = self.bdim
        if isinstance(index, tuple):
            for i in index[: self.bdim]:
                bdim -= is_0d(i)
        elif is_0d(index):
            bdim -= 1
        if bdim > 0:
            return self.__class__(self.data[index], bdim=bdim)
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __iter__(self):
        return iter(self.data)

    def append(self, x):
        self.extend(np.asarray(x)[None, ...])

    def find(self, value, first=True):
        matches = self.data == value

        # remove extra dimensions
        for _ in range(matches.ndim - self.bdim):
            matches = np.all(np.asarray(matches), axis=-1)

        index = np.argwhere(matches)[:, 0]

        if first:
            # take only the first match:
            index = None if len(index) == 0 else index[0]

        return index

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def reshape(self, *shape, index=None, bdim=None):
        if index is not None:
            assert index.step is None or index.step == 1
            shape = self.shape[: index.start] + shape + self.shape[index.stop :]

        if bdim is None:
            bdim = compute_bdim(self.shape, self.bdim, shape)

        return self.__class__(reshape(self.data, shape), bdim=bdim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        import torch as tr
        types = tuple(tr.Tensor if issubclass(t, ArrayDataset) else t for t in types)
        args = tuple(tr.as_tensor(a.data) if isinstance(a, ArrayDataset) else a for a in args)
        kwargs = (
            None
            if kwargs is None
            else {k: tr.as_tensor(a.data) if isinstance(a, ArrayDataset) else a for k, a in kwargs.items()}
        )
        return tr.Tensor.__torch_function__(func, types, args, kwargs)


class NumpyDataset(ArrayDataset):
    """Dataset with Numpy array as data."""

    def __init__(self, data, **kwargs):
        super().__init__(np.asarray(data), **kwargs)

    def extend(self, X):
        self.data = np.append(self.data, np.asarray(X), axis=0)

    def astype(self, dtype, copy=False, convert_torch=True):
        if convert_torch and not isinstance(dtype, np.dtype):
            from ..tumpy.torch_bindings import dtype_t_to_n
            dtype = dtype_t_to_n.get(dtype, dtype)
        data = self.data.astype(dtype, copy=copy)
        if copy:
            return self.__class__(data, bdim=self.bdim)
        else:
            self.data = data
            return self

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype, copy=False)


class TorchDataset(ArrayDataset):
    """Dataset with torch.tensor as data."""

    def extend(self, X):
        import torch

        if isinstance(X, Dataset):
            X = X.data
        self.data = torch.cat((self.data, torch.tensor(X)), axis=0)

    def astype(self, dtype, copy=False, convert_numpy=True):
        if convert_numpy and isinstance(dtype, np.dtype):
            from ..tumpy.torch_bindings import dtype_t_to_n

            dtype = dtype_t_to_n.get(dtype, dtype)
        data = self.data.type(dtype)
        if copy:
            data = data.clone() if data is self.data else data
            return self.__class__(data, bdim=self.bdim)
        else:
            self.data = data
            return self

    def __array__(self, dtype=None):
        data = self.data.numpy()
        return data if dtype is None else data.astype(dtype, copy=False)


class ObjectDataset(NumpyDataset):
    """Dataset with variable entries"""

    def __init__(self, data, **kwargs):
        """Init to cast data to np.ndarray with dtype=object"""
        data = np.asarray(data, dtype=object)
        super().__init__(data, has_Xy=False, **kwargs)

    @property
    def dtype(self):
        """Property used to check if dataset is sequence dataset"""
        return object

    def _bad__getitem__(self, index):
        if isinstance(index, tuple):
            index, k = index[: self.bdim], index[self.bdim :]
        else:
            k = ()
        data = self.data[index]
        bdim = self.bdim + getattr(data, "ndim", 0) - self.data.ndim

        if len(k):
            data = np.array([r[k] for r in data])

        if bdim:
            return self.__class__(data, bdim=bdim)
        else:
            return data

    def astype(self, dtype, **kwargs):
        if dtype == object:
            return self
        raise TypeError(f"Can't convert `dtype` from `object` to {dtype}")

    @property
    def shape(self):
        #try:
        #    feat_shape = self.data[0].shape
        #except AttributeError:
        #    feat_shape = (None,)
        return *self.data.shape, None

    def reshape(self, *shape, index=None, **kwargs):
        """Think I can make this work..."""
        raise NotImplementedError("Reshape is not supported by ObjectDataset due to variable-length inputs.")

    def reshape_batch(self, *shape, index=None, **kwargs):
        if index is not None:
            assert index.step is None or index.step == 1
            shape = self.data.shape[: index.start] + shape + self.data.shape[index.stop :]

        return self.__class__(self.data.reshape(shape), bdim=len(shape))

    def swapaxes(self, a0, a1, **kwargs):
        assert a0 != 0 and a1 != 0
        a0 = a0 - 1 if a0 > 0 else a0
        a1 = a1 - 1 if a1 > 0 else a1
        return self.__class__([row.swapaxes(a0, a1) for row in self.data])


class PandasSeriesDataset(NumpyDataset):
    def __init__(self, data, **kwargs):
        super().__init__(data.values, **kwargs)


class DictDataset(TeachableWrapperDataset):
    """
    Contains a dictionary whose values are datasets.

    For indexing purposes, the first `self.bdim` axes (i.e., the
    batch dimensions) index into the first axes of the constituent
    datasets, whereas the dictionary key "dimension" occurs right after
    the batch dimensions. Since there is usually exactly one batch
    dimension, this means you can index like

    >>> dataset[:20, 'X']

    which will return the first 20 rows of the `'X'` constituent dataset,
    whereas

    >>> dataset[:20]

    will take the first 20 rows of each constituent dataset, and package
    them into a new `DictDataset` with the same keys.
    """

    def __init__(self, data={}, convert_sequences=True, bdim=1, has_Xy=None, **kw_data):  # NOSONAR
        data = update_copy(data, kw_data)  # NOSONAR
        super().__init__(None, bdim=bdim, has_Xy=bool({"X", "x", "y"} & set(data)) if has_Xy is None else has_Xy)
        self.data = {
            k: TeachableDataset.from_data(d, recursive=False, convert_sequences=convert_sequences, bdim=bdim)
            for k, d in data.items()
        }

    def append(self, x):
        for key in self.data.keys():
            self.data[key].append(x[key])

    def extend(self, X):
        if isinstance(X, DictDataset):
            X = X.data
        for key in self.data.keys():
            self.data[key].extend(X[key])

    def reshape(self, *shape, index=None, bdim=None):
        if bdim is None:
            if index is not None:
                assert isinstance(index, slice)
                new_shape = self.shape[: index.start] + shape + self.shape[index.stop :]
            else:
                new_shape = shape
            bdim = compute_bdim(self.shape, self.bdim, new_shape)

        if shape[bdim] != len(self.data):
            raise ValueError(
                "When reshaping a DictDataset, the first non-batch dimension must equal the number of keys."
            )
        shape = shape[:bdim] + shape[bdim + 1 :]

        return self.__class__({k: reshape(v, shape, index) for k, v in self.data.items()}, bdim=bdim)

    def _get_bdim(self, i, k, sub_data, test_key):
        try:
            bdim = self.bdim + sub_data[test_key].ndim - self.data[test_key].ndim
        except AttributeError:
            bdim = self.bdim - len(i)
            for j in i:
                if isinstance(j, (MutableSequence, slice)) or hasattr(k, "__array__"):
                    bdim += 1
        return bdim

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) > self.bdim:
            # i is the indices into each dataset in the dictionary
            i = index[: self.bdim] + index[self.bdim + 1 :]

            # k is the dict key(s)
            k = index[self.bdim]

            if k == slice(None, None):
                k = self.data.keys()
            elif not (isinstance(k, MutableSequence) or hasattr(k, "__array__")):
                # single dict key, so return its value
                return self.data[k][i]

        else:
            i = (index,)
            k = self.data.keys()

        sub_data = {key: self.data[key][i] for key in k}
        test_key = next(iter(k))
        bdim = self._get_bdim(i, k, sub_data, test_key)

        if bdim == 0:  # batch is fully-indexed, so we return a dict
            return sub_data
        else:  # some batch indices remain, so return a DictDataset
            return self.__class__(sub_data, bdim=bdim)

    def __setitem__(self, index, value):
        raise NotImplementedError

    def __iter__(self):
        for i in np.ndindex(self.shape[: self.bdim]):
            yield {k: v[i] for k, v in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __setattr__(self, name, value):
        if name in {"data", "bdim", "has_Xy"} or name[:2] == "__":
            object.__setattr__(self, name, value)
        else:
            self.data[name] = value

    def __getattr__(self, name):
        try:
            if name in {"data", "bdim", "has_Xy"} or name[:2] == "__":
                return object.__getattr__(self, name)
            else:
                return self.data[name]
        except (IndexError, TypeError, KeyError):
            raise AttributeError

    def find(self, value, first=True):
        indices = tuple(self.data[k].find(value[k], first=False) for k in value.keys())
        while len(indices) > 1:
            indices = (
                np.intersect1d(indices[0], indices[1], assume_unique=True),
                *(indices[2:]),
            )
        index = indices[0]
        if first:
            index = None if len(index) == 0 else index[0]
        return index

    @property
    def X(self):
        try:
            return self.data["X"]
        except KeyError:
            raise AttributeError("No 'X' key in DictDataset.")

    @property
    def y(self):
        try:
            return self.data["y"]
        except KeyError:
            raise AttributeError("No 'y' key in DictDataset.")

    @property
    def shape(self):
        inner_shape = next(iter(self.data.values())).shape
        return inner_shape[: self.bdim] + (len(self.data),) + inner_shape[self.bdim :]

    @property
    def ndim(self):
        return next(iter(self.data.values())).ndim + 1

    @property
    def dtype(self):
        dts = [v.dtype for v in self.data.values()]
        if not all([d == dts[0] for d in dts]):
            raise TypeError(
                "To call self.dtype, all subdtypes must be equal.\n"
                "In this case, the subdtypes are:\n"
                f"{[(k, v.dtype) for k, v in self.data.items()]}"
            )
        return dts[0]

    def astype(self, dtype, copy=False):
        data = self.data if not copy else data.copy()
        for k in data:
            data[k] = data[k].astype(dtype, copy=copy)
        if copy:
            return self.__class__(data, bdim=self.bdim)
        else:
            return self


class DataFrameDataset(DictDataset):
    def __init__(self, data, **kwargs):
        import pandas as pd

        assert isinstance(data, pd.DataFrame)
        super().__init__({k: data[k].values for k in data.columns}, **kwargs)


class TupleDataset(TeachableWrapperDataset):
    """Dataset with Tuple as self.data."""

    def __init__(self, data, convert_sequences=True, bdim=1):
        super().__init__(None, bdim=bdim)
        self.data = tuple(
            TeachableDataset.from_data(d, recursive=False, convert_sequences=convert_sequences, bdim=bdim) for d in data
        )

    def append(self, x):
        for data_n, x_n in zip(self.data, x):
            data_n.append(x_n)

    def extend(self, X):
        for data_n, x_n in zip(self.data, X):
            data_n.extend(x_n)

    def reshape(self, *shape, index=None, bdim=None):
        if bdim is None:
            self_shape = self.data[0].shape
            if index is not None:
                assert isinstance(index, slice)
                new_shape = self_shape[: index.start] + shape + self_shape[index.stop :]
            else:
                new_shape = shape
            bdim = compute_bdim(self_shape, self.bdim, new_shape)

        if shape[bdim] != len(self.data):
            raise ValueError(
                "When reshaping a TupleDataset, the first non-batch dimension must equal the number of keys."
            )
        shape = shape[:bdim] + shape[bdim + 1 :]

        return self.__class__(tuple(reshape(v, shape, index) for v in self.data), bdim=bdim)

    def __getitem__(self, index):
        # Case 1: indexing multiple axes
        if isinstance(index, tuple) and len(index) > self.bdim:
            # i is the indices into each dataset in the tuple
            i = index[: self.bdim] + index[self.bdim + 1 :]

            # k is the tuple key(s)
            k = index[self.bdim]

            if is_0d(k):
                # returning a single dataset in the tuple
                return self.data[k][i]
            elif isinstance(k, slice):
                # select a slice of the tuple
                sub_data = tuple(d[i] for d in self.data[k])
            # else:
            #     # selecting multiple elements of the tuple
            #     sub_data = tuple(d[key][i] for key in k) # TODO: d is undefined here

        else:
            sub_data = tuple(d[index] for d in self.data)

        bdim = getattr(sub_data[0], "bdim", 0)
        if bdim == 0:  # batch is fully-indexed, so we return a tuple
            return sub_data
        else:  # some batch indices remain, so return a TupleDataset
            return self.__class__(sub_data, bdim=bdim)

    def __iter__(self):
        return zip(*(self.data))

    def __len__(self):
        return len(self.data[0])

    def __array__(self, dtype=None):
        arrays = list(np.asarray(X_n, dtype=dtype) for X_n in self.data)

        max_dim = max(a.ndim for a in arrays)
        for i, arr in enumerate(arrays):
            while arr.ndim < max_dim:
                arr = np.expand_dims(arr, 1)
            arrays[i] = arr
        # concatenate = False  # TODO: adhoc fix, `concatenate` variable was missing here

        # if concatenate:
        #     return np.concatenate(arrays, axis=1)
        # else:
        #     return np.stack(arrays, axis=1)
        return np.stack(arrays, axis=1)

    def find(self, value, first=True):
        indices = tuple(d_n.find(v_n, first=False) for d_n, v_n in zip(self.data, value))
        while len(indices) > 1:
            indices = (
                np.intersect1d(indices[0], indices[1], assume_unique=True),
                *(indices[2:]),
            )
        index = indices[0]
        if first:
            index = None if len(index) == 0 else index[0]
        return index

    @property
    def tuple(self):
        """Getter for self.data."""
        return self.data

    @property
    def shape(self):
        inner_shape = self.data[0].shape
        return inner_shape[: self.bdim] + (len(self.data),) + (inner_shape[self.bdim :])

    @property
    def X(self):
        raise AttributeError("New: TupleDatasets no longer have an `X` attribute.")
        X = self.data[:-1]
        return TupleDataset(X) if len(X) > 1 else X[0]

    @property
    def y(self):
        raise AttributeError("New: TupleDatasets no longer have a `y` attribute.")
        return self.data[-1]

    @property
    def dtype(self):
        dts = tuple(v.dtype for v in self.data)
        if not all([d == dts[0] for d in dts]):
            raise TypeError(
                f"To call self.dtype, all subdtypes must be equal.\n" f"In this case, the subdtypes are:\n{dts}"
            )
        return dts[0]

    def astype(self, dtype, copy=False):
        data = tuple(d.astype(dtype, copy=copy) for d in self.data)
        if copy:
            return self.__class__(data, bdim=self.bdim)
        else:
            self.data = data
            return self
