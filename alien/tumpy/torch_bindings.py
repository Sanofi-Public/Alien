import math

import numpy as np
import torch as tr


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, tr.Tensor):
        # if (dtype is None or data.dtype == dtype) and (data.device is None or data.device == device):
        #    return data
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return tr._as_tensor(data, dtype=dtype, device=device)
    elif "tensorflow" in str(type(data)):
        return tr._as_tensor(data.numpy(), dtype=dtype, device=device)
    return tr._as_tensor(data, dtype=dtype, device=device)


tr._as_tensor = tr.as_tensor
tr.as_tensor = as_tensor

# TODO: allow setting default dtype, default device

dtype_n_to_t = {
    float: tr.float,
    int: tr.int,
    bool: tr.bool,
    # np.bool       : tr.bool,
    np.uint8: tr.uint8,
    np.int8: tr.int8,
    np.int16: tr.int16,
    np.int32: tr.int32,
    np.int64: tr.int64,
    np.float16: tr.float16,
    np.float32: tr.float32,
    np.float64: tr.float64,
    np.complex64: tr.complex64,
    np.complex128: tr.complex128,
    np.integer: np.integer,
    np.floating: np.floating,
}

dtype_t_to_n = {v: k for k, v in dtype_n_to_t.items()}


class Tumpy:

    bool_ = bool
    uint8 = tr.uint8
    int8 = tr.int8
    int16 = tr.int16
    int32 = tr.int32
    int64 = tr.int64
    float16 = tr.float16
    float32 = tr.float32
    float64 = tr.float64
    complex64 = tr.complex64
    complex128 = tr.complex128

    integer = np.integer
    floating = np.floating

    # Constants
    inf = math.inf
    e = math.e
    pi = math.pi
    tau = math.tau
    nan = math.nan

    # Functions

    def arange(*start, dtype=None, like=None, device=None):
        return TumpyTensor(
            tr.arange(
                *start, dtype=dtype_n_to_t.get(dtype, dtype), device=Tumpy.default_device if device is None else device
            )
        )

    def array(data, dtype=None, copy=True, order="K", subok=False, ndmin=0, like=None, device=None):  # NOSONAR
        if device is None:
            device = Tumpy.default_device
        try:
            return TumpyTensor(tr.tensor(data, dtype=dtype_n_to_t.get(dtype, dtype)))
        except (TypeError, ValueError) as e:
            try:
                return np.array(
                    data,
                    dtype=dtype_t_to_n.get(dtype, dtype),
                    copy=copy,
                    order=order,
                    subok=subok,
                    ndmin=ndmin,
                    like=like,
                )
            except:
                raise e

    def asarray(data, dtype=None, order=None, like=None, device=None):
        if device is None:
            device = Tumpy.default_device
        try:
            return TumpyTensor(tr.as_tensor(data, dtype=dtype_n_to_t.get(dtype, dtype), device=device))
        except (TypeError, ValueError) as e:
            try:
                return np.asarray(data, dtype=dtype_t_to_n.get(dtype, dtype), order=order)
            except:
                raise e
            
    def as_tensor(data, dtype=None, device=None):
        return tr.as_tensor(
            data, 
            dtype=dtype_n_to_t.get(dtype, dtype), 
            device=device
        ).as_subclass(tr.Tensor)

    def zeros(shape, dtype=float, order="C", like=None, device=None):
        if device is None:
            device = Tumpy.default_device
        return TumpyTensor(
            tr.zeros((shape,) if isinstance(shape, int) else shape, dtype=dtype_n_to_t.get(dtype, dtype), device=device)
        )

    def count_nonzero(a, axis=None, keepdims=False):
        return TumpyTensor(tr.count_nonzero(tr.as_tensor(a), axis, keepdim=keepdims))

    def empty(shape, dtype=float, order="C", like=None, device=None):
        if device is None:
            device = Tumpy.default_device
        try:
            return TumpyTensor(
                tr.empty(
                    (shape,) if isinstance(shape, int) else shape, dtype=dtype_n_to_t.get(dtype, dtype), device=device
                )
            )
        except (TypeError, ValueError) as e:
            try:
                return np.empty(shape, dtype=dtype_t_to_n.get(dtype, dtype), order=order)
            except:
                raise e

    # Couldn't implement numpy.dtype

    def frombuffer(buffer, dtype=float, count=-1, offset=0, like=None):
        return TumpyTensor(tr.frombuffer(buffer, dtype=dtype_n_to_t.get(dtype, dtype), count=count, offset=offset))

    def from_dlpack(x):
        return tr.from_dlpack(tr.as_tensor(x))

    def fromiter(i, dtype, count=-1, device=None):
        if device is None:
            device = Tumpy.default_device
        return TumpyTensor(tr.as_tensor(np.fromiter(i, dtype_t_to_n.get(dtype, dtype), count=count), device=device))

    # Couldn't implement numpy.where

    def argwhere(a):
        return TumpyTensor(r := tr.argwhere(tr.as_tensor(a)))

    def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
        return TumpyTensor(tr.cat([tr.as_tensor(a) for a in arrays], axis, out=out))

    def delete(arr, obj, axis=None):
        if axis is None:
            arr = arr.flatten()
            axis = 0
        assert isinstance(obj, int), "Deletion of fancy indices not yet implemented."
        i_a = tr.arange(arr.shape[axis] - 1, device=arr.device)
        i_a[obj:] += 1
        arr = tr.as_tensor(arr)
        return TumpyTensor(tr.index_select(tr.as_tensor(arr), axis, i_a))
        # return arr.take(tr.arange(arr.shape[axis]) != obj, dim=axis)

    # Couldn't implement numpy.can_cast

    # Couldn't implement numpy.promote_types

    # Couldn't implement numpy.result_type

    def empty_like(prototype, dtype=None, order="K", subok=True, shape=None, device=None):
        return TumpyTensor(tr.empty_like(tr.asarray(prototype), dtype=dtype_n_to_t.get(dtype, dtype), device=device))

    def zeros_like(a, dtype=None, order="K", subok=True, shape=None, device=None):
        return TumpyTensor(tr.zeros_like(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype), device=device))

    def ones_like(a, dtype=None, order="K", subok=True, shape=None, device=None):
        return TumpyTensor(tr.ones_like(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype), device=device))

    def inner(a, b):
        return TumpyTensor(tr.inner(tr.as_tensor(a), tr.as_tensor(b)))

    def dot(a, b, out=None):
        return TumpyTensor(tr.mm(tr.as_tensor(a), tr.as_tensor(b), out=out))

    def outer(a, b, out=None):
        return TumpyTensor(tr.outer(tr.as_tensor(a), tr.as_tensor(b), out=out))

    def vdot(a, b):
        return TumpyTensor(tr.vdot(tr.as_tensor(a), tr.as_tensor(b)))

    def roll(a, shift, axis=None):
        return TumpyTensor(tr.roll(tr.as_tensor(a), shift, axis))

    def moveaxis(a, source, destination):
        return TumpyTensor(tr.moveaxis(tr.as_tensor(a), source, destination))

    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return TumpyTensor(tr.cross(tr.as_tensor(a), tr.as_tensor(b), axis))

    def tensordot(a, b, axes=2):
        return tr.tensordot(tr.as_tensor(a), tr.as_tensor(b), dims=axes)

    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return TumpyTensor(tr.isclose(tr.as_tensor(a), tr.as_tensor(b), rtol, atol, equal_nan))

    def ones(shape, dtype=None, order="C", like=None, device=None):
        if device is None:
            device = Tumpy.default_device
        return TumpyTensor(
            tr.ones((shape,) if isinstance(shape, int) else shape, dtype=dtype_n_to_t.get(dtype, dtype), device=device)
        )

    def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return tr.allclose(tr.as_tensor(a), tr.as_tensor(b), rtol, atol, equal_nan)

    def invert(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.bitwise_not(tr.as_tensor(x), out=out))

    bitwise_not = invert

    def full(shape, fill_value, dtype=None, order="C", like=None, device=None):
        if device is None:
            device = Tumpy.default_device
        return TumpyTensor(
            tr.full(
                (shape,) if isinstance(shape, int) else shape,
                fill_value,
                dtype=dtype_n_to_t.get(dtype, dtype),
                device=device,
            )
        )

    def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
        return TumpyTensor(tr.full_like(a, fill_value, dtype=dtype))

    def matmul(
        x1,
        x2,
        out=None,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
        axes=None,
        axis=None,
    ):
        return TumpyTensor(tr.matmul(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def all(a, axis=None, out=None, keepdims=False, where=None):
        if axis is None or a.ndim < 2:
            return tr.all(tr.as_tensor(a)).item()
        return TumpyTensor(tr.all(tr.as_tensor(a), axis, out=out, keepdim=keepdims))

    def amax(a, axis=None, out=None, keepdims=False):
        if axis is None:
            a = a.flatten()
            axis = 0
        # if a.ndim < 2 and not keepdims:
        #    return tr.amax(tr.as_tensor(a), keepdim=keepdims, out=out).item()
        return TumpyTensor(tr.amax(tr.as_tensor(a), axis, keepdim=keepdims, out=out))

    max = amax

    def amin(a, axis=None, out=None, keepdims=False, initial=None, where=None):
        if axis is None:
            a = a.flatten()
            axis = 0
        if a.ndim < 2 and not keepdims:
            return tr.amin(tr.as_tensor(a), keepdim=keepdims, out=out).item()
        return TumpyTensor(tr.amin(tr.as_tensor(a), axis, keepdim=keepdims, out=out))

    min = amin

    def any(a, axis=None, out=None, keepdims=False, where=None):
        if axis is None or a.ndim < 2:
            return tr.any(tr.as_tensor(a)).item()
        return TumpyTensor(tr.any(tr.as_tensor(a), axis, out=out, keepdim=keepdims))

    def argmax(a, axis=None, out=None, keepdims=False):
        return (
            tr.argmax(tr.as_tensor(a)).item()
            if axis is None or a.ndim < 2
            else TumpyTensor(tr.argmax(tr.as_tensor(a), axis, keepdims))
        )

    def argmin(a, axis=None, out=None, keepdims=False):
        return TumpyTensor(tr.argmin(tr.as_tensor(a), axis, keepdims))

    def argsort(a, axis=-1, kind=None, order=None):
        return TumpyTensor(tr.argsort(tr.as_tensor(a), axis))

    def clip(a, a_min, a_max, out, kwargs):
        return TumpyTensor(tr.clip(tr.as_tensor(a), a_min, a_max, out=out))

    def cumprod(a, axis=None, dtype=None, out=None):
        return TumpyTensor(tr.cumprod(tr.as_tensor(a), axis, dtype=dtype_n_to_t.get(dtype, dtype), out=out))

    def cumsum(a, axis=None, dtype=None, out=None):
        if axis is None:
            a = a.flatten()
            axis = 0
        return TumpyTensor(tr.cumsum(tr.as_tensor(a), axis, dtype=dtype_n_to_t.get(dtype, dtype), out=out))

    def diagonal(a, offset=0, axis1=0, axis2=1):
        return TumpyTensor(tr.diagonal(tr.as_tensor(a), offset, dim1=axis1, dim2=axis2))

    def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=None):
        if axis is None or a.ndim < 2:
            return tr.mean(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype)).item()
        return TumpyTensor(
            tr.mean(tr.as_tensor(a), axis, dtype=dtype_n_to_t.get(dtype, dtype), out=out, keepdim=keepdims)
        )

    # Couldn't implement numpy.nonzero

    def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
        if axis is None or a.ndim < 2:
            return tr.prod(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype)).item()
        else:
            return TumpyTensor(tr.prod(tr.as_tensor(a), axis, dtype=dtype_n_to_t.get(dtype, dtype), keepdim=keepdims))

    # Couldn't implement numpy.put

    def ravel(a, order="C"):
        return TumpyTensor(tr.ravel(tr.as_tensor(a)))

    def reshape(a, newshape, order="C"):
        return TumpyTensor(tr.reshape(tr.as_tensor(a), (newshape,) if isinstance(newshape, int) else newshape))

    def round_(a, decimals=0, out=None):
        return TumpyTensor(tr.round(tr.as_tensor(a), decimals=decimals, out=out))

    round = round_

    def searchsorted(a, v, side="left", sorter=None):
        return TumpyTensor(tr.searchsorted(tr.as_tensor(a), v, side=side, sorter=sorter))

    def sort(a, axis=-1, kind=None, order=None):
        return tr.sort(tr.as_tensor(a), axis)

    def squeeze(a, axis=None):
        return TumpyTensor(tr.squeeze(tr.as_tensor(a), axis))

    def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=None):
        return TumpyTensor(tr.std(tr.as_tensor(a), axis, out=out, keepdim=keepdims))

    def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
        ret = tr.sum(tr.as_tensor(a), axis, dtype=dtype_n_to_t.get(dtype, dtype), out=out, keepdim=keepdims)
        return ret.item() if axis is None else TumpyTensor(ret)

    def swapaxes(a, axis1, axis2):
        return TumpyTensor(tr.swapaxes(tr.as_tensor(a), axis2, axis1))

    def take(a, indices, axis=None, *, out=None):
        if axis is None:
            return TumpyTensor(tr.take(tr.as_tensor(a), indices))
        return TumpyTensor(tr.index_select(tr.as_tensor(a), axis, indices))

    def trace(a, offset=0, axis1=0, axis2=1, dtype=None):  # , out=None
        ret = tr.sum(tr.diagonal(tr.as_tensor(a), offset=offset, dim1=axis1, dim2=axis2), dim=-1)
        return TumpyTensor(ret) if isinstance(ret, tr.Tensor) and ret.ndim > 0 else ret.item()

    def transpose(a, axes=None):
        if axes is None:
            return TumpyTensor(a.T)
        return TumpyTensor(tr.Tensor.transpose(tr.as_tensor(a), *axes))

    def var_v2(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=None):
        if axis is None:
            a = a.flatten()
            axis = 0
            keepdims = False
        return TumpyTensor(tr.var(tr.as_tensor(a), axis, keepdim=keepdims, out=out, correction=ddof))

    def var_v1(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=None):
        if axis is None:
            a = a.flatten()
            axis = 0
            keepdims = False
        return TumpyTensor(tr.var(tr.as_tensor(a), axis, bool(ddof), keepdims, out=out))

    def absolute(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.abs(tr.as_tensor(x), out=out))

    abs = absolute

    def add(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.add(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def arccos(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.acos(tr.as_tensor(x), out=out))

    def arccosh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.arccosh(tr.as_tensor(x), out=out))

    def arcsin(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.asin(tr.as_tensor(x), out=out))

    def arcsinh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.arcsinh(tr.as_tensor(x), out=out))

    def arctan(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.atan(tr.as_tensor(x), out=out))

    def arctan2(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.arctan2(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def arctanh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.arctanh(tr.as_tensor(x), out=out))

    def bitwise_and(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.bitwise_and(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def bitwise_or(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.bitwise_or(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def bitwise_xor(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.bitwise_xor(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def ceil(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.ceil(tr.as_tensor(x), out=out))

    def conjugate(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.conj(tr.as_tensor(x)))

    conj = conjugate

    def copysign(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.copysign(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def cos(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.cos(tr.as_tensor(x), out=out))

    def cosh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.cosh(tr.as_tensor(x), out=out))

    def deg2rad(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.deg2rad(tr.as_tensor(x), out=out))

    def divide(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.true_divide(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    true_divide = divide

    def equal(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return tr.equal(tr.as_tensor(x1), tr.as_tensor(x2))

    def exp(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.exp(tr.as_tensor(x), out=out))

    def exp2(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.exp2(tr.as_tensor(x), out=out))

    def expm1(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.expm1(tr.as_tensor(x), out=out))

    def softmax(x, axis=None):
        return TumpyTensor(tr.nn.functional.softmax(tr.as_tensor(x), dim=axis))

    def floor(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.floor(tr.as_tensor(x), out=out))

    def floor_divide(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.floor_divide(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def power(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.power(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def float_power(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.float_power(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def fmax(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.fmax(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def fmin(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.fmin(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def fmod(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.fmod(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    # Couldn't implement numpy.frexp

    def gcd(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.gcd(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def greater(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.greater(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def greater_equal(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.greater_equal(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def heaviside(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.heaviside(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def hypot(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.hypot(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def isfinite(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.isfinite(tr.as_tensor(x)))

    def isinf(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.isinf(tr.as_tensor(x)))

    def isnan(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.isnan(tr.as_tensor(x)))

    def lcm(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.lcm(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def ldexp(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.ldexp(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def less(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.less(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def less_equal(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.less_equal(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def log(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.log(tr.as_tensor(x), out=out))

    def log10(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.log10(tr.as_tensor(x), out=out))

    def log1p(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.log1p(tr.as_tensor(x), out=out))

    def log2(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.log2(tr.as_tensor(x), out=out))

    def logaddexp(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.logaddexp(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def logaddexp2(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.logaddexp2(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def logical_and(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.logical_and(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def logical_not(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.logical_not(tr.as_tensor(x), out=out))

    def logical_or(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.logical_or(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def logical_xor(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.logical_xor(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def maximum(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.maximum(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def minimum(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.minimum(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def multiply(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return tr.multiply(tr.as_tensor(x1), tr.as_tensor(x2), out=out)

    def negative(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.negative(tr.as_tensor(x), out=out))

    def nextafter(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.nextafter(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def not_equal(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.not_equal(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def positive(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.positive(tr.as_tensor(x)))

    def rad2deg(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.rad2deg(tr.as_tensor(x), out=out))

    def reciprocal(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.reciprocal(tr.as_tensor(x), out=out))

    def remainder(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.remainder(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    mod = remainder

    def sign(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.sign(tr.as_tensor(x), out=out))

    def signbit(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.signbit(tr.as_tensor(x), out=out))

    def sin(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.sin(tr.as_tensor(x), out=out))

    def sinh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.sinh(tr.as_tensor(x), out=out))

    def sqrt(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.sqrt(tr.as_tensor(x), out=out))

    def square(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.square(tr.as_tensor(x), out=out))

    def subtract(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None,
    ):
        return TumpyTensor(tr.subtract(tr.as_tensor(x1), tr.as_tensor(x2), out=out))

    def tan(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.tan(tr.as_tensor(x), out=out))

    def tanh(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.tanh(tr.as_tensor(x), out=out))

    def trunc(
        x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True, signature=None, extobj=None
    ):
        return TumpyTensor(tr.trunc(tr.as_tensor(x), out=out))

    def set_printoptions(
        precision=None,
        threshold=None,
        edgeitems=None,
        linewidth=None,
        suppress=None,
        nanstr=None,
        infstr=None,
        formatter=None,
        sign=None,
        floatmode=None,
        legacy=None,
    ):
        return tr.set_printoptions(precision, threshold, edgeitems, linewidth)

    def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
        return TumpyTensor(tr.logspace(start, stop, num, base, dtype=dtype_n_to_t.get(dtype, dtype)))

    def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, device=None):
        return TumpyTensor(tr.linspace(start, stop, num, dtype=dtype_n_to_t.get(dtype, dtype), device=device))

    # Couldn't implement numpy.finfo

    # Couldn't implement numpy.iinfo

    def atleast_1d(arys):
        return tr.atleast_1d(arys)

    def atleast_2d(arys):
        return tr.atleast_2d(arys)

    def atleast_3d(arys):
        return tr.atleast_3d(arys)

    def hstack(tup):
        return TumpyTensor(tr.hstack(tup))

    def stack(arrays, axis=0, out=None):
        return TumpyTensor(tr.stack([tr.as_tensor(a) for a in arrays], axis, out=out))

    def vstack(tup):
        return TumpyTensor(tr.row_stack(tup))

    row_stack = vstack

    def einsum(operands, out, optimize, kwargs):
        return tr.einsum(operands)

    def imag(val):
        return TumpyTensor(tr.imag(val))

    def isreal(x):
        return TumpyTensor(tr.isreal(tr.as_tensor(x)))

    def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
        return TumpyTensor(tr.nan_to_num(tr.as_tensor(x), nan, posinf, neginf))

    def real(val):
        return TumpyTensor(tr.real(val))

    def typename(char):
        return tr.typename(char)

    # Couldn't implement numpy.select

    def diff(a, n=1, axis=-1, prepend=None, append=None):
        return TumpyTensor(tr.diff(tr.as_tensor(a), n, axis, prepend, append))

    # Couldn't implement numpy.gradient

    def angle(z, deg=False):
        return TumpyTensor(tr.angle(z))

    def flip(m, axis=None):
        if axis is None:
            axis = list(range(m.ndim))
        elif isinstance(axis, int):
            axis = [axis]
        return TumpyTensor(tr.flip(m, dims=axis))

    # Couldn't implement numpy.rot90

    def bincount(x, weights=None, minlength=0):
        return TumpyTensor(tr.bincount(tr.as_tensor(x), weights, minlength))

    def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
        m = tr.as_tensor(m)
        if y is not None:
            m = tr.cat((m, tr.as_tensor(y)))
        if not rowvar:
            m = m.T
        return TumpyTensor(
            tr.cov(tr.tensor(m), correction=1 - bias if ddof is None else ddof, fweights=fweights, aweights=aweights)
        )

    def corrcoef(x, y=None, rowvar=True, bias=None, ddof=None, dtype=None):
        return TumpyTensor(tr.corrcoef(tr.as_tensor(x)))

    def msort(a):
        return TumpyTensor(tr.msort(tr.as_tensor(a)))

    def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
        return (
            tr.median(tr.as_tensor(a).flatten()).item()
            if axis is None or a.ndim < 2
            else TumpyTensor(tr.median(tr.as_tensor(a), axis, keepdims, out=out)[0])
        )

    def sinc(x):
        return TumpyTensor(tr.sinc(tr.as_tensor(x)))

    def trapz(y, x=None, dx=1.0, axis=-1):
        return TumpyTensor(tr.trapz(y, x, dim=axis))

    def i0(x):
        return TumpyTensor(tr.i0(tr.as_tensor(x)))

    def meshgrid(xi, copy=True, sparse=False, indexing="xy"):
        return tr.meshgrid(xi, indexing=indexing)

    def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, interpolation=None):
        return TumpyTensor(tr.quantile(tr.as_tensor(a), q, axis, interpolation=interpolation, out=out, keepdim=keepdims))

    def column_stack(tup):
        return TumpyTensor(tr.column_stack(tup))

    def dstack(tup):
        return TumpyTensor(tr.dstack(tup))

    def split(ary, indices_or_sections, axis=0):
        return tr.split(ary, indices_or_sections, axis)

    # Couldn't implement numpy.hsplit

    # Couldn't implement numpy.vsplit

    # Couldn't implement numpy.dsplit

    def kron(a, b):
        return TumpyTensor(tr.kron(tr.as_tensor(a), tr.as_tensor(b)))

    def tile(A, reps):
        return TumpyTensor(tr.tile(A, reps))

    def broadcast_to(array, shape, subok=False):
        return TumpyTensor(tr.broadcast_to(tr.as_tensor(array), (shape,) if isinstance(shape, int) else shape))

    def broadcast_shapes(*args):
        return tr.broadcast_shapes(*args)

    def broadcast_arrays(*args):
        return tuple(TumpyTensor(t) for t in tr.broadcast_tensors(*args))

    def diag(v, k=0):
        return TumpyTensor(tr.diag(v))

    def diagflat(v, k=0):
        return TumpyTensor(tr.diagflat(v))

    def eye(N, M=None, k=0, dtype=float, order="C", like=None):
        return TumpyTensor(tr.eye(N, dtype=dtype_n_to_t.get(dtype, dtype)))

    def fliplr(m):
        return TumpyTensor(tr.fliplr(m))

    def flipud(m):
        return TumpyTensor(tr.flipud(m))

    def triu(m, k=0):
        return TumpyTensor(tr.triu(m))

    def tril(m, k=0):
        return TumpyTensor(tr.tril(m, k))

    def vander(x, N=None, increasing=False):
        return TumpyTensor(tr.vander(tr.as_tensor(x), N, increasing))

    def tril_indices(n, k=0, m=None):
        return (i := tr.tril_indices(n, n if m is None else m, k))[0], i[1]

    # Couldn't implement numpy.triu_indices

    def fix(x, out=None):
        return TumpyTensor(tr.fix(tr.as_tensor(x), out=out))

    def isneginf(x, out=None):
        return TumpyTensor(tr.isneginf(tr.as_tensor(x), out=out))

    def isposinf(x, out=None):
        return TumpyTensor(tr.isposinf(tr.as_tensor(x), out=out))

    def unique(x, *, return_inverse=False, return_counts=False, axis=None):
        rv = tr.unique(tr.as_tensor(x), return_inverse=return_inverse, return_counts=return_counts, dim=axis)
        return tuple(TumpyTensor(t) for t in rv) if (return_inverse or return_counts) else TumpyTensor(rv)

    def isin(element, test_elements, assume_unique=False, invert=False):
        try:
            element = tr.as_tensor(element)
            return TumpyTensor(
                tr.isin(element, tr.as_tensor(test_elements), assume_unique=assume_unique, invert=invert)
            )
        except:
            np.isin(element, test_elements, assume_unique=assume_unique, invert=invert)

    def issubdtype(arg1, arg2):
        return np.issubdtype(dtype_t_to_n.get(arg1, arg1), dtype_t_to_n.get(arg2, arg2))

    def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding="ASCII", max_header_size=10000):
        return tr.load(file)

    def save(file, arr, allow_pickle=True, fix_imports=True):
        return tr.save(file, tr.as_tensor(arr))

    def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
        if axis is None or a.ndim < 2:
            return tr.nansum(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype)).item()
        return TumpyTensor(tr.nansum(tr.as_tensor(a), axis, keepdims, dtype=dtype_n_to_t.get(dtype, dtype), out=out))

    def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=None):
        if axis is None or a.ndim < 2:
            return tr.nanmean(tr.as_tensor(a), dtype=dtype_n_to_t.get(dtype, dtype)).item()
        return TumpyTensor(tr.nanmean(tr.as_tensor(a), axis, keepdims, dtype=dtype_n_to_t.get(dtype, dtype), out=out))

    def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
        if axis is None or a.ndim < 2:
            return tr.nanmedian(tr.as_tensor(a), dtype=dtype_n_to_t.get(a.dtype, a.dtype)).item()
        return TumpyTensor(
            tr.nanmedian(tr.as_tensor(a), axis, keepdims, dtype=dtype_n_to_t.get(a.dtype, a.dtype), out=out)
        )

    def nanquantile(
        a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, interpolation=None
    ):
        return TumpyTensor(tr.nanquantile(tr.as_tensor(a), q, axis, interpolation=interpolation, out=out, keepdim=keepdims))

    def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
        return tr.histogram(tr.as_tensor(a), bins, range=range, density=density)

    def histogramdd(sample, bins=10, range=None, normed=None, weights=None, density=None):
        return tr.histogramdd(tr.as_tensor(sample), bins, range=range, density=density)

    def ndindex(*shape):
        return TumpyTensor(tr.as_tensor(np.ndindex(*shape)))
        # return tr.stack(tr.meshgrid(*(tr.arange(n) for n in shape))).reshape(len(shape), -1).T

    def to(a, device):
        if device is None:
            return TumpyTensor(tr.as_tensor(a))
        if isinstance(device, tr.Tensor):
            device = device.device
        return TumpyTensor(tr.as_tensor(a, device=device))

    @staticmethod
    def no_grad():
        return tr.no_grad()

    def is_float(x):
        if isinstance(x, np.ndarray):
            return x.dtype.kind not in "iu"
        return tr.is_floating_point(x) or tr.is_complex(x)

    def is_integer(x):
        return not Tumpy.is_float(x)

    def is_bool(x):
        return x.dtype in {bool, tr.bool}

    def device(x):
        return tr.as_tensor(x).device

    def is_array(x):
        return isinstance(x, tr.Tensor) or isinstance(x, np.ndarray)

    class linalg:
        def matrix_power(a, n):
            return TumpyTensor(tr.linalg.matrix_power(tr.as_tensor(a), n))

        def solve(a, b):
            return TumpyTensor(tr.linalg.solve(tr.as_tensor(a), tr.as_tensor(b)))

        # Couldn't implement numpy.ndarray.tensorsolve

        def tensorinv(a, ind=2):
            return TumpyTensor(tr.linalg.tensorinv(tr.as_tensor(a), ind))

        def inv(a):
            return TumpyTensor(tr.linalg.inv(tr.as_tensor(a)))

        def pinv(a, rcond=1e-15, hermitian=False):
            return TumpyTensor(tr.linalg.pinv(tr.as_tensor(a), rtol=rcond, hermitian=hermitian))

        def cholesky(a):
            return TumpyTensor(tr.linalg.cholesky(tr.as_tensor(a)))

        def eigvals(a):
            return TumpyTensor(tr.linalg.eigvals(tr.as_tensor(a)))

        def eigvalsh(a, UPLO="L"):
            return TumpyTensor(tr.linalg.eigvalsh(tr.as_tensor(a), UPLO))

        # Couldn't implement numpy.ndarray.pinv

        def slogdet(a):
            return tr.linalg.slogdet(tr.as_tensor(a))

        def det(a):
            return TumpyTensor(tr.linalg.det(tr.as_tensor(a)))

        def svd(a, full_matrices=True, compute_uv=True):
            r = tr.linalg.svd(tr.as_tensor(a), full_matrices=full_matrices)
            if compute_uv:
                return TumpyTensor(r[0]), TumpyTensor(r[1]), TumpyTensor(r[2])
            else:
                return TumpyTensor(r[1])

        def eig(a):
            return tr.linalg.eig(tr.as_tensor(a))

        def eigh(a, UPLO="L"):
            return tr.linalg.eigh(tr.as_tensor(a), UPLO)

        def lstsq(a, b, rcond="warn"):
            return tr.linalg.lstsq(tr.as_tensor(a), tr.as_tensor(b), rcond)

        def norm(a, ord=None, axis=None, keepdims=False):
            return tr.linalg.norm(tr.as_tensor(a), ord, axis, keepdims)

        def qr(a, mode="reduced"):
            return tr.linalg.qr(tr.as_tensor(a), mode)

        def cond(x, p=None):
            return TumpyTensor(tr.linalg.cond(tr.as_tensor(x), p))

        # Couldn't implement numpy.ndarray.matrix_rank

        # Couldn't implement numpy.ndarray.LinAlgError

        def multi_dot(arrays, out=None):
            return tr.linalg.multi_dot([tr.as_tensor(a) for a in arrays], out=out)

    class random:
        def default_rng(seed=None):
            return RNG(seed)

    default_device = None

    def set_default_device(device):
        if device == "gpu":
            device = "cuda"
        if str(device)[:4] == "cuda" and not tr.cuda.is_available():
            raise ValueError("Tried to set device to 'cuda', but no CUDA is available.")
        Tumpy.default_device = device

    @staticmethod
    def get_default_device():
        return Tumpy.default_device


class TumpyTensor(tr.Tensor):
    def __new__(cls, x):
        return x.as_subclass(cls)

    def __array__(self):
        return tr.detach(self).cpu().numpy()

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return str(self.__array__())

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        types = tuple(tr.Tensor if t is TumpyTensor else t for t in types)
        args = tuple(a.as_subclass(tr.Tensor) if isinstance(a, TumpyTensor) else a for a in args)
        kwargs = (
            None
            if kwargs is None
            else {k: v.as_subclass(tr.Tensor) if isinstance(v, TumpyTensor) else v for k, v in kwargs.items()}
        )
        ret = tr.Tensor.__torch_function__(func, types, args, kwargs)
        if isinstance(ret, tr.Tensor):
            return TumpyTensor(ret)
        elif isinstance(ret, tuple):
            return tuple(TumpyTensor(r) if isinstance(r, tr.Tensor) else r for r in ret)
        return ret

    @property
    def dtype(self):
        sdt = super().dtype
        return dtype_t_to_n.get(sdt, sdt)

    def all(self, axis=None, out=None, keepdims=False, where=None):
        if axis is None or self.ndim < 2:
            return tr.all(self).item()
        return TumpyTensor(tr.all(self, axis, keepdims, out=out))

    def any(self, axis=None, out=None, keepdims=False, where=None):
        if axis is None or self.ndim < 2:
            return tr.any(self).item()
        return TumpyTensor(tr.any(self, axis, keepdims, out=out))

    def astype(self, dtype):
        return TumpyTensor(self.to(dtype=dtype_n_to_t.get(dtype, dtype)))

    def argmax(self, axis=None, out=None, keepdims=False):
        return tr.argmax(self).item() if axis is None or self.ndim < 2 else TumpyTensor(tr.argmax(self, axis, keepdims))

    def argmin(self, axis=None, out=None, keepdims=False):
        return tr.argmin(self).item() if axis is None or self.ndim < 2 else TumpyTensor(tr.argmin(self, axis, keepdims))

    def argsort(self, axis=-1, kind=None, order=None):
        return tr.Tensor.argsort(self, axis)

    def clip(self, min=None, max=None, out=None):
        return TumpyTensor(tr.Tensor.clip(self, min, max))

    def conj(self):
        return TumpyTensor(tr.Tensor.conj(self))

    def copy(self):
        return TumpyTensor(tr.Tensor.clone(self))

    def cumprod(self, axis=None, dtype=None, out=None):
        return TumpyTensor(tr.Tensor.cumprod(self, axis, dtype=dtype_n_to_t.get(dtype, dtype)))

    def cumsum(self, axis=None, dtype=None, out=None):
        return TumpyTensor(tr.Tensor.cumsum(self, axis, dtype=dtype_n_to_t.get(dtype, dtype)))

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return TumpyTensor(tr.Tensor.diagonal(self, offset, dim1=axis1, dim2=axis2))

    # Couldn't implement numpy.ndarray.dot

    def flatten(self, order="C"):
        return TumpyTensor(tr.Tensor.flatten(self))

    def item(self):  # , args):
        return tr.Tensor.item(self)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        ret = tr.Tensor.amax(tr.as_tensor(self), axis, out=out, keepdim=keepdims)
        return ret.item() if axis is None else TumpyTensor(ret)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, where=True):
        if axis is None or self.ndim < 2:
            return tr.Tensor.mean(self, dtype=dtype_n_to_t.get(dtype, dtype)).item()
        return TumpyTensor(tr.Tensor.mean(self, axis, keepdims, dtype=dtype_n_to_t.get(dtype, dtype)))

    def min(self, axis, out, keepdims):#, initial, where=True):
        return TumpyTensor(tr.Tensor.amin(self, axis, out=out, keepdim=keepdims))

    def nonzero(self):
        return tr.Tensor.nonzero(self)

    def norm(self, ord=None, axis=None, keepdims=False):
        return tr.Tensor.norm(self, ord, axis, keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        if axis is None or self.ndim < 2:
            return tr.Tensor.prod(self, dtype=dtype_n_to_t.get(dtype, dtype)).item()
        return TumpyTensor(tr.Tensor.prod(self, axis, keepdims, dtype=dtype_n_to_t.get(dtype, dtype)))

    # Couldn't implement numpy.ndarray.put

    def ravel(self, order=None):
        return TumpyTensor(tr.Tensor.ravel(self))

    def repeat(self, repeats):
        return TumpyTensor(tr.Tensor.repeat(self, *repeats))

    def reshape(self, *shape, order="C"):
        return TumpyTensor(tr.Tensor.reshape(self, *(shape[0] if isinstance(shape[0], tuple) else shape)))

    def resize(self, new_shape, refcheck=True):
        return TumpyTensor(tr.Tensor.resize(self, new_shape))

    def round(self, decimals=0, out=None):
        return TumpyTensor(tr.Tensor.round(self, decimals))

    def sort(self, axis=-1, kind=None, order=None):
        return tr.Tensor.sort(self, axis)

    def squeeze(self, axis=None):
        return TumpyTensor(tr.Tensor.squeeze(self) if axis is None else tr.Tensor.squeeze(self, axis))

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
        return TumpyTensor(tr.Tensor.std(self, axis, correction=ddof, keepdim=keepdims))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        ret = tr.sum(tr.as_tensor(self), axis, dtype=dtype_n_to_t.get(dtype, dtype), out=out, keepdim=keepdims)
        return ret.item() if axis is None else TumpyTensor(ret)

    def swapaxes(self, axis1, axis2):
        return TumpyTensor(tr.Tensor.swapaxes(self, axis2, axis1))

    def take(self, indices):
        return TumpyTensor(tr.Tensor.take(self, indices))

    def tolist(self):
        return tr.Tensor.tolist(self)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        ret = tr.sum(tr.diagonal(tr.as_tensor(self), offset=offset, dim1=axis1, dim2=axis2), dim=-1)
        return TumpyTensor(ret) if isinstance(ret, tr.Tensor) and ret.ndim > 0 else ret.item()

    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(0, self.ndim, -1))
        return TumpyTensor(tr.Tensor.transpose(self, *axes))

    def var_v2(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=None):
        if axis is None:
            self = self.flatten()
            axis = 0
            keepdims = False
        return TumpyTensor(tr.var(self, axis, keepdim=keepdims, out=out, correction=ddof))

    def var_v1(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=None):
        if axis is None:
            self = self.flatten()
            axis = 0
            keepdims = False
        return TumpyTensor(tr.var(self, axis, bool(ddof), keepdims, out=out))

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return TumpyTensor(super().view(*shape))


NumpyTensor = TumpyTensor


for name, fn in Tumpy.__dict__.copy().items():
    if tr.__version__ < "2" and name.endswith("_v1"):
        setattr(Tumpy, name[:-3], fn)
    elif name.endswith("_v2"):
        setattr(Tumpy, name[:-3], fn)

for name, fn in TumpyTensor.__dict__.copy().items():
    if tr.__version__ < "2" and name.endswith("_v1"):
        setattr(TumpyTensor, name[:-3], fn)
    elif name.endswith("_v2"):
        setattr(TumpyTensor, name[:-3], fn)


class RNG:
    def __init__(self, seed=None, device=None):
        if device is None:
            device = Tumpy.default_device
        self.gen = tr.Generator(device=device)
        # super().__init__(device=device)
        # tr.Generator.__init__(tr.as_tensor(self), device=device)
        if seed and seed != 0:  # isinstance(seed, (int, np.integer, tr.int32, tr.int64)):
            self.gen.manual_seed(int(seed))

    def integers(self, high: int, low: int = 0, size: tuple = (), dtype=np.int64, endpoint: bool = False):
        if low > high:
            low, high = high, low
        if endpoint:
            high += 1
        result = TumpyTensor(
            tr.randint(
                low=int(low),
                high=int(high),
                size=(size,) if isinstance(size, int) else size,
                dtype=dtype_n_to_t.get(dtype, dtype),
                generator=self.gen,
            )
        )
        if size == ():
            result = result.item()
        return result

    def random(self, size=None, dtype=np.int64, out=None):
        return TumpyTensor(
            tr.rand(
                *(size,) if isinstance(size, int) else size,
                dtype=dtype_n_to_t.get(dtype, dtype),
                generator=self.gen,
            )
        )

    def normal(self, loc=0.0, scale=1.0, size=1):
        size = (size,) if isinstance(size, int) else size
        return TumpyTensor(
            tr.normal(
                tr.broadcast_to(tr.as_tensor(loc), size),
                tr.broadcast_to(tr.as_tensor(scale), size),
                generator=self.gen,
            )
        )

    def uniform(self, low=0.0, high=1.0, size=1, dtype=np.float32):
        return low + self.random(size, dtype=dtype_n_to_t.get(dtype, dtype)) * (high - low)

    def multivariate_normal(self, mean, cov, size=1, check_valid="warn", tol=1e-8, method="svd"):
        #breakpoint()
        if True:  # (dist := getattr(tr.as_tensor(self), 'multivariate_normal_dist', None)) is None:
            self.multivariate_normal_dist = (
                dist := tr.distributions.multivariate_normal.MultivariateNormal(Tumpy.as_tensor(mean), Tumpy.as_tensor(cov))
            )
        return TumpyTensor(dist.rsample((size,) if isinstance(size, int) else size))

    def choice(self, a, size=1, replace=True, p=tr.tensor(1.0), axis=0, shuffle=True):
        assert isinstance(size, int), "Higher-dim sizes not implemented yet."
        assert axis == 0, "Selection from axis != 0 not implemented yet."
        if isinstance(a, int):
            array = None
            shape = (a,)
        else:
            array = a
            shape = array.shape

        p = tr.broadcast_to(p, shape)
        indices = p.multinomial(
            num_samples=size,
            replacement=replace,
            generator=self.gen,
        )

        return TumpyTensor(indices) if array is None else array[indices]

    def shuffle(self, x, axis=0):
        x[:] = tr.index_select(tr.as_tensor(x), axis, tr.randperm(x.shape[axis], generator=self.gen))

    def permutation(self, x, axis=0):
        if isinstance(tr.as_tensor(x), int):
            return TumpyTensor(tr.randperm(tr.as_tensor(x), generator=self.gen))
        else:
            return TumpyTensor(tr.index_select(x, axis, tr.randperm(x.shape[axis], generator=self.gen)))
