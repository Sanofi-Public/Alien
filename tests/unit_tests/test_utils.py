import numpy as np
import pytest

from alien.utils import (
    SelfDict,
    add_slice,
    axes_except,
    chunks,
    concatenate,
    dict_pop,
    isint,
    reshape,
    seed_all,
    zip_dict,
    any_get,
    any_pop,
)


def test_seed_all():
    seed_all(0)


def test_zip_dict():
    zipped = zip_dict({"a": 1}, {"a": 2})
    assert list(zipped.keys()) == ["a"]
    assert set(list(zipped["a"])) == {1, 2}
    with pytest.raises(KeyError):
        zipped = zip_dict({"a": 1}, {"b": 2})
        list(zipped["a"])


def test_isint():
    test = 1
    assert isint(test)
    test = 1.0
    assert isint(test)
    test = 1.5
    assert not isint(test)
    test = "a"
    assert not isint(test)
    test = np.zeros(1)
    assert isint(test)
    test = np.zeros(2)
    assert not isint(test)


def test_axes_except():
    x = np.zeros((2, 3, 5))
    assert axes_except(x, 0) == (1, 2)
    assert axes_except(x, (0, 1)) == (2,)


def test_dict_pop():
    test_dict = {"a": 1, "b": 2, "c": 3}
    out_dict = dict_pop(test_dict, "d")
    assert out_dict == {}, "dict_pop should return empty key for empty "
    assert test_dict == {"a": 1, "b": 2, "c": 3}, "dict_pop shouldn't pop non-existing key"
    out_dict = dict_pop(test_dict, "c")
    assert "c" in out_dict
    assert "c" not in test_dict
    out_dict = dict_pop(test_dict, b=3)
    assert "b" in out_dict and out_dict["b"] == 2
    assert "b" not in test_dict


class TestSelfDict:
    # pylint: disable=no-member
    self_dict = SelfDict()
    self_dict["a"] = 1
    self_dict["b"] = 2

    def test_set(self):
        self_dict = TestSelfDict.self_dict
        self_dict["c"] = 3
        assert self_dict.c == 3

    def test_attr(self):
        self_dict = TestSelfDict.self_dict
        assert self_dict.a == 1

    def test_pop(self):
        self_dict = TestSelfDict.self_dict
        out = self_dict.pop("a")
        assert out == 1
        assert "a" not in self_dict

        out = self_dict.pop(["b"])
        assert out.b == 2
        assert "b" not in self_dict


def test_chunks():
    seq = list(range(8))
    chunk_size = 3
    c_1 = chunks(seq, chunk_size)
    exp = [[0, 1, 2], [3, 4, 5], [6, 7]]
    for i, x in enumerate(c_1):
        assert exp[i] == x
    assert len(c_1) == int(np.ceil(len(seq) / chunk_size))


def test_reshape():
    n_rows, n_cols = 13, 23
    X_np = np.zeros(shape=(n_rows, n_cols))
    reshaped = reshape(X_np, shape=(n_cols, n_rows))
    assert reshaped.shape == X_np.T.shape
    with pytest.raises(ValueError):
        _ = reshape(X_np, shape=(1,))
    reshaped = reshape(X_np, shape=(n_cols * n_rows,))
    assert reshaped.shape == (n_cols * n_rows,)

    torch = pytest.importorskip("torch")
    X_torch = torch.zeros(n_rows, n_cols)
    reshaped = reshape(X_torch, shape=(n_cols, n_rows))
    assert reshaped.shape == X_torch.T.shape

    tf = pytest.importorskip("tensorflow")
    X_tf = tf.zeros(shape=(n_rows, n_cols))
    reshaped = reshape(X_tf, shape=(n_cols, n_rows))
    assert reshaped.shape == tf.transpose(X_tf).shape


def test_add_slice():
    lst = list(range(11))
    i = 3
    slice_1 = slice(2, 5)
    slice_2 = add_slice(slice_1, i)
    assert all([lst[slice_1][x] + i == lst[slice_2][x] for x in range(len(lst[slice_1]))])

    slice_2 = add_slice(None, i)
    assert lst[slice_2] == lst[i:]

    slice_1 = slice(None, None, 2)
    slice_2 = add_slice(slice_1, i)
    assert lst[slice_2] == lst[i:][slice_1]


def test_concatenate():
    torch = pytest.importorskip("torch")
    shape = (2, 3)
    arr_1 = np.zeros(shape)
    arr_2 = torch.zeros(shape)
    with pytest.raises(TypeError):
        concatenate(arr_1, arr_2)
    assert (concatenate(arr_2, arr_2) == torch.cat((arr_2, arr_2), dim=0)).all()

def test_any_pop():
    d = {'x':0, 'y':1, 'z':2}
    s = set(d.keys())
    keys = ['y','x']
    assert any_pop(d, keys) == 1
    assert any_pop(d, keys) == 0
    with pytest.raises(KeyError):
        assert any_pop(d, keys)
    assert any_pop(d, keys, None) is None

    assert any_pop(s, keys) == 'y'
    assert any_pop(s, keys) == 'x'
    with pytest.raises(KeyError):
        assert any_pop('z', keys)



