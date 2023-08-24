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
from collections.abc import MutableSequence
from typing import Any, Optional, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike

if "torch" in sys.modules:
    import torch

from ..utils import add_slice, reshape, isint, update_copy


class Dataset(metaclass=ABCMeta):
    """
    Abstract interface to a readable dataset.
    """

    def __new__(cls, *args, **kwargs):
        if cls == Dataset:
            return Dataset.from_data(*args, **kwargs)
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
    def X(self):
        """Return features."""
        self.check_Xy()
        if self.bdim == 1:
            return Dataset.from_data(self[:, :-1], recursive=False)
        else:
            i = (slice(None),) * self.bdim + (slice(None, -1),)
            return Dataset.from_data(self[i], recursive=False)

    @property
    def y(self):
        """Return targets."""
        self.check_Xy()
        if self.bdim == 1:
            return Dataset.from_data(self[:, -1], recursive=False)
        else:
            i = (slice(None),) * self.bdim + (-1,)
            return Dataset.from_data(self[i], recursive=False)
    
    def check_Xy(self):
        if not self.has_Xy:
            warnings.warn("Dataset doesn't store separate `X` or `y` columns.")

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
                TeachableDataset.from_data(
                    data, recursive=False, convert_sequences=convert_sequences, **kwargs
                ),
                shuffle=shuffle,
                random_seed=random_seed,
            )
        elif data is None or isinstance(data, dict):
            return DictDataset(data, convert_sequences=convert_sequences, **kwargs)
        elif convert_sequences and isinstance(data, MutableSequence):
            return NumpyDataset(np.asarray(data), **kwargs)
        elif isinstance(data, TeachableDataset) or isinstance(data, MutableSequence):
            return TeachableWrapperDataset(data, **kwargs) if recursive else data
        elif isinstance(data, np.ndarray):
            return NumpyDataset(data, **kwargs)
        elif isinstance(data, tuple):
            return TupleDataset(data, convert_sequences=convert_sequences, **kwargs)
        elif "torch" in str(type(data)):
            return TorchDataset(data, **kwargs)
        elif "DataFrame" in str(type(data)):
            return DictDataset({k: data[k].values for k in data.columns})
        elif "deepchem" in str(type(data)):
            return TeachableDataset.from_deepchem(data)
        else:
            warnings.warn("Passing an unknown data format into TeachableDataset.")
            return TeachableWrapperDataset(data)

    @staticmethod
    def from_deepchem(data):
        try:
            # pylint: disable=import-outside-toplevel
            import deepchem

            assert isinstance(data, deepchem.data.Dataset)
            from .deepchem import DeepChemDataset

            return DeepChemDataset(data)
        except Exception as exc:
            raise NotImplementedError(
                "We thought this was a DeepChem dataset, but apparently not!"
            ) from exc

    def get_shuffle(self, shuffle="random", random_seed=None):
        """Return a shuffled version of self

        Args:
            shuffle (str, optional): The initial shuffle - `'identity'` or `'random'`. Defaults to `'random`'.
            random_seed (int, optional): A random seed for the shuffle. Defaults to None.

        Returns:
            ShuffledDataset: A shuffled version of `self`
        """
        return ShuffledDataset(self, shuffle=shuffle, random_seed=random_seed)


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


class ShuffledDataset(TeachableWrapperDataset):
    """
    Presents a shuffle of an existing dataset (or MutableSequence)

    Added data goes at the end and isn't shuffled (until reshuffle() is called).

    :param data: the existing dataset to wrap
    :param shuffle: determines the initial shuffle state: 'random' or 'identity'
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
        assert bdim == 1, "ShuffledDataset is only possible with one batch dimension."
        super().__init__(data)
        self.rng = np.random.default_rng(random_seed)
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

    def reshuffle(
        self,
        # random_seed: Optional[
        #     Union[int, ArrayLike, SeedSequence, BitGenerator, Generator]
        # ] = None,
    ):
        """Reshuffles self with self.rng."""
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
        return iter(TeachableDataset.from_data(self.data[self.shuffle]))

    def __array__(self, dtype=None):
        "Converts to a Numpy array"
        return np.array(self.data, dtype=dtype)[self.shuffle]

    @property
    def X(self):
        X = ShuffledDataset(self.data.X, shuffle=self.shuffle)
        X.rng = None
        return X

    @property
    def y(self):
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
        if isint(index):
            bdim -= 1
        elif isinstance(index, tuple):
            for i in index[: self.bdim]:
                bdim -= isint(i)
        if bdim > 0:
            return self.__class__(self.data[index], bdim=bdim)
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def append(self, x):
        self.extend(np.array(x)[None, ...])

    def find(self, value, first=True):
        matches = self.data == value

        # remove extra dimensions
        for _ in range(matches.ndim - self.bdim):
            matches = np.all(np.array(matches), axis=-1)

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


class NumpyDataset(ArrayDataset):
    """Dataset with Numpy array as data."""

    def extend(self, X):
        self.data = np.append(self.data, np.asarray(X), axis=0)

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype, copy=False)


class TorchDataset(ArrayDataset):
    """Dataset with torch.tensor as data."""

    def extend(self, X):
        import torch

        if isinstance(X, Dataset):
            X = X.data
        self.data = torch.cat((self.data, torch.tensor(X)), axis=0)

    def __array__(self, dtype=None):
        data = self.data.numpy(force=True)
        return data if dtype is None else data.astype(dtype, copy=False)


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

    def __init__(self,
        data={}, # NOSONAR
        convert_sequences=True,
        bdim=1,
        has_Xy=None,
        **kw_data
    ):
        data = update_copy(data, kw_data) # NOSONAR
        super().__init__(None, bdim=bdim,
            has_Xy=bool({'X','x','y'} & set(data)) if has_Xy is None else has_Xy)
        self.data = {
            k: TeachableDataset.from_data(
                d, recursive=False, convert_sequences=convert_sequences, bdim=bdim
            )
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
            raise ValueError("When reshaping a DictDataset, the first non-batch dimension must equal the number of keys.")
        shape = shape[:bdim] + shape[bdim+1:]

        return self.__class__(
            {k: reshape(v, shape, index) for k, v in self.data.items()}, bdim=bdim
        )

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) > self.bdim:
            # i is the indices into each dataset in the dictionary
            i = index[: self.bdim] + index[self.bdim + 1 :]

            # k is the dict key(s)
            k = index[self.bdim]

            if k == slice(None, None):
                k = self.data.keys()
            elif not isinstance(k, MutableSequence):
                # single dict key, so return its value
                return self.data[k][i]

        else:
            i = index
            k = self.data.keys()

        sub_data = {key: self.data[key][i] for key in k}

        bdim = getattr(next(iter(sub_data.values())), "bdim", 0)
        if bdim == 0:  # batch is fully-indexed, so we return a dict
            return sub_data
        else:  # some batch indices remain, so return a DictDataset
            return self.__class__(sub_data, bdim=bdim)

    def __setitem__(self, index, value):
        raise NotImplementedError

    def __iter__(self):
        for i in np.ndindex(self.shape[:self.bdim]):
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
        self.check_Xy()
        return self.data["X"]

    @property
    def y(self):
        self.check_Xy()
        return self.data["y"]

    @property
    def shape(self):
        inner_shape = next(iter(self.data.values())).shape
        return inner_shape[: self.bdim] + (len(self.data),) + inner_shape[self.bdim :]

    @property
    def ndim(self):
        return next(iter(self.data.values())).ndim + 1


class TupleDataset(TeachableWrapperDataset):
    """Dataset with Tuple as self.data."""

    def __init__(self, data, convert_sequences=True, bdim=1):
        super().__init__(None, bdim=bdim)
        self.data = tuple(
            TeachableDataset.from_data(
                d, recursive=False, convert_sequences=convert_sequences, bdim=bdim
            )
            for d in data
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
            raise ValueError("When reshaping a TupleDataset, the first non-batch dimension must equal the number of keys.")
        shape = shape[:bdim] + shape[bdim+1:]

        return self.__class__(tuple(reshape(v, shape, index) for v in self.data), bdim=bdim)

    def __getitem__(self, index):
        # Case 1: indexing multiple axes
        if isinstance(index, tuple) and len(index) > self.bdim:
            # i is the indices into each dataset in the tuple
            i = index[: self.bdim] + index[self.bdim + 1 :]

            # k is the tuple key(s)
            k = index[self.bdim]

            if isint(k):
                # returning a single dataset in the tuple
                return self.data[k][i]
            elif isinstance(k, slice):
                # select a slice of the tuple
                sub_data = tuple(d[i] for d in self.data[k])
            else:
                # selecting multiple elements of the tuple
                # TODO: d is undefined here
                sub_data = tuple(d[key][i] for key in k)

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

        if concatenate:
            return np.concatenate(arrays, axis=1)
        else:
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
        return inner_shape[:self.bdim] + (len(self.data),) + (inner_shape[self.bdim:])

    @property
    def X(self):
        self.check_Xy()
        X = self.data[:-1]
        return TupleDataset(X) if len(X) > 1 else X[0]

    @property
    def y(self):
        self.check_Xy()
        return self.data[-1]
