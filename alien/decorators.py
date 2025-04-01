from functools import partial, wraps
from inspect import Parameter, signature
from typing import List, Optional, Union

import numpy as np

from .utils import flatten, isint, reshape


def default_decorator(dec):
    """
    Enhances a parametrized decorator by giving it the following
    functionality:
    
    If the decorator is applied directly to a function, without any
    parameters specified, then it decorates the function in the usual
    way, with default values for the parameters.
    
    If the decorator is passed parameters, then it returns a new
    decorator (with the given parameter values), able to accept a
    function to decorate.
    
    In other words, if we have::
    
        @default_decorator
        decorator(fn, a=1, b=2, ...):
            ...

    then::    

        @decorator
        def fn(...):
            ...
    
    gives the usual behaviour for a decorator, but::

        @decorator(3, b=4)
        def fn(...):
            ...
    
    gives us the function::

        decorator(fn, a=3, b=4)
    """
    argnames = list(signature(dec).parameters)

    def wrapped_dec(*args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            return dec(*args, **kwargs)
        else:
            kwargs.update(zip(argnames[1:], args))
            return partial(dec, **kwargs)

    return wrapped_dec


@default_decorator
def flatten_batch(fn, bdim=0, to_flatten=1, is_method=None, keep_self_dim=True, degree=1):  # NOSONAR
    """
    Decorator which 'flattens' batch dimensions, and
    restores their shape upon return. Used as a convenience
    by functions which need (or want) to assume there is exactly
    one batch dimension.

    Note: If the first flattenable input has a `bdim` attribute, this
    decorator simply flattens those dimensions, plus or minus `bdim`,
    ignoring all of the other complicated calculations detailed below.
    This should produce relatively consistent behaviour.

    :param bdim: number of batch dimensions to flatten.
        If negative, keeps that many final dimensions
        unflattened, i.e., the batch dimensions is
            `args[0].ndim + bdim`.
        If `is_method` and `keep_self_dim` are both True, then
        `bdim` modifies the batch dimension computed from
        self.ndim and args[0].ndim (as explained in `keep_self_dim`).

    :param keep_self_dim: If True (and if `is_method` is True also),
        computes the batch dimensions as the difference
            `args[0].ndim - self.ndim + bdim`.

    :param is_method: If True, treat the first argument as `self`, and
        count other arguments from the second position. If `is_method`
        is True as well as `keep_self_dim`, this triggers the
        batch dimension calculation explained in `keep_self_dim`.
        By default, `is_method = (argnames[0] == 'self')`.

    :param to_flatten: number of arguments (not including 'self' if
        `fn` is a method) to be flattened in this way. You can also
        provide a sequence of argument names or numbers. Defaults to 1.

    :param degree: tensor degree of the output over the batch dimensions.
        Concretely, `degree` specifies how many dimensions of the raw
        output should be reshaped into copies of the batch shape.
        Defaults to 1, which is correct if the function provides an
        output for each sample in the batch. 0 would be correct if, eg.,
        the function aggregates all the samples into a summary statistic.
        2 would be correct if, eg., the function outputs a matrix over
        the samples.
    """
    sig = signature(fn)
    argnames = list(sig.parameters)
    if is_method is None:
        is_method = argnames[0] == "self"

    if is_method:
        a0 = argnames[1]
    else:
        a0 = argnames[0]
        keep_self_dim = False

    to_flatten = range(to_flatten) if isint(to_flatten) else to_flatten
    to_flatten = [argnames[n + is_method] if isint(n) else n for n in to_flatten]

    @wraps(fn)
    def flat_fn(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        args = bound_args.arguments

        b_dim = args.pop("bdim", bdim)  # If the end-user passes a bdim value, use that instead

        if b := getattr(args[a0], "bdim", None) is not None:
            b_dim += b
        elif not hasattr(args[a0], "shape"):
            b_dim = 1
        elif is_method and keep_self_dim:
            self = bound_args.args[0]
            b_dim += args[a0].ndim - self.ndim
            assert b_dim <= args[a0].ndim, f"Computed batch ndim {b_dim} is greater than input ndim {args[a0].ndim}."
        elif b_dim <= 0:
            b_dim += args[a0].ndim

        # shortcut for most common case:
        if b_dim == 1:
            return fn(*bound_args.args, **bound_args.kwargs)

        bshape = args[a0].shape[:b_dim]

        try:
            args.update((n, flatten(args[n], b_dim)) for n in to_flatten)
        except NotImplementedError:
            return fn(*bound_args.args, **bound_args.kwargs)

        result = fn(*bound_args.args, **bound_args.kwargs)

        if isinstance(result, tuple):
            return tuple(reshape(v, degree * bshape + v.shape[degree:]) for v in result)
        return reshape(result, degree * bshape + result.shape[degree:])

    return flat_fn


def get_args(other_fn=None, exclude={}):
    argnames = set(signature(other_fn).parameters) - set(exclude)
    from utils import dict_pop

    def get_args_decorator(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            params = dict_pop(kwargs, *argnames)
            return fn(*args, params=params, **kwargs)

        return wrapped_fn

    return get_args_decorator


def do_normalize(X, bdim=-1, euclidean=False):
    bdims = tuple(range(bdim)) if bdim >= 0 else tuple(range(X.ndim + bdim))
    X_std = np.std(np.array(X), axis=bdims)
    X_std[X_std == 0] = 1
    if euclidean:
        X_std *= np.sqrt(np.prod(X.shape[:bdim]))
    return X / X_std


def sig_append(fn, name, kind=Parameter.KEYWORD_ONLY, default=Parameter.empty, annotation=Parameter.empty):
    """
    Appends a new parameter to the signature of `fn`.
    """
    sig = signature(fn)
    fn.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())
        + [
            Parameter(
                name,
                kind,
                default=default,
                annotation=annotation,
            )
        ]
    )
    return fn


@default_decorator
def normalize_args(fn, to_norm=1, bdim=None, euclidean=False, keep_self_dim=True):
    """
    Decorator that normalizes the argument(s) to a function, in each
    dimension.

    :param to_norm: number of arguments (after 'self') to
        normalize. May also be a list of integers and strings (for
        keyword args). Defaults to 1.
    :param bdim: number of batch dimensions to normalize over. May be negative,
        in which case it specified how many of the last dimensions *not* to
        normalize over. If `keep_self_dim` is `True`, then this number is added
        to the number of batch dimensions determined by excluding `self.ndim`.

        Defaults to -1 if keep_self_dim is False or `fn` isn't a method, else
        defaults to 0 (keeping exactly `self.ndim` dimensions).
    :param keep_self_dim: if True, keeps `self.ndim` final dimensions, and normalizes
        over the others. In other words, the number of batch dimensions will be
            X.ndim - self.ndim + bdim
    :param euclidean: whether to divide by sqrt(N) (N is the size
        of the feature dimension). Should be true if you'll be using
        Euclidean distances. Defaults to False.
    """
    sig = signature(fn)
    argnames = list(sig.parameters)
    is_method = _check_is_method(argnames)

    to_norm = range(to_norm) if isint(to_norm) else to_norm
    to_norm = [_get_arg_name(n, argnames, is_method) for n in to_norm]

    pass_normalize = "normalize" in argnames

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        normalize = kwargs.get("normalize", True)
        if not pass_normalize and "normalize" in kwargs:
            del kwargs["normalize"]

        bound_args = sig.bind(*args, **kwargs)
        args = bound_args.arguments

        b_dim = _get_batch_dim(args, is_method, bdim, keep_self_dim)

        if normalize:
            for a in to_norm:
                args[a] = do_normalize(args[a], bdim=b_dim, euclidean=euclidean)

        return fn(*bound_args.args, **bound_args.kwargs)

    if not pass_normalize:
        # add `normalize` parameter to wrapped signature:
        sig_append(wrapped_fn, "normalize", Parameter.KEYWORD_ONLY, default=True, annotation=bool)

    return wrapped_fn


def _check_is_method(argnames):
    """Helper function to check whether call to normalize_args is from a class method or function."""
    return argnames[0] == "self" if argnames else False


def _get_arg_name(arg: Union[int, str], argnames: List[str], is_method: bool) -> str:
    """Helper for normalize_args to get argument name."""
    if isint(arg):
        return argnames[arg + is_method]
    else:
        return arg


def _get_batch_dim(args: dict, is_method: bool, batch_dim: Optional[int], keep_self_dim: bool) -> int:
    """Helper function to get batch dimension for normalize_args."""
    if is_method and keep_self_dim:
        return -args["self"].ndim if batch_dim is None else batch_dim - args["self"].ndim
    else:
        return -1 if batch_dim is None else batch_dim


def get_Xy(fn):  # NOSONAR
    """
    This decorator ensures that a method gets appropriate `X` and `y`
    values. A method decorated in this way can assume that its `X` (or `x`)
    and `y` arguments have nontrivial values, drawn from an appropriate
    source.

    If both `X` (or `x`) and `y` are passed in (Nones don't count),
    then all's well. If neither are passed in, then this wrapper looks for
    them in `self.X` and `self.y`.

    If an `X` value can be found, but no `y` value, this wrapper treats
    `X` as combined `data`, and tries to split it into `data.X` and `data.y`.
    Failing that, it tries to take `X = data[:,:-1]` and `y = data[:,-1]`.

    If none of this can be accomplished, this wrapper raises a `ValueError`.
    """
    sig = signature(fn)
    names = list(sig.parameters)

    X_name, y_name = _get_xy_names(names)

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        self = args[0]
        args = bound_args.arguments

        if X_name not in args or args[X_name] is None:
            try:
                args[X_name] = getattr(self, X_name)
                if hasattr(self, y_name):
                    args[y_name] = getattr(self, y_name, None)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"You didn't pass any data to `{fn.__name__}`, and we couldn't find it in `self`.")

        if y_name not in args or args[y_name] is None:
            data = args[X_name]
            try:
                args[X_name] = data.X
                args[y_name] = data.y
            except (KeyError, AttributeError):
                try:
                    args[X_name] = data[:, :-1]
                    args[y_name] = data[:, -1]
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"You didn't pass a separate y-value, but we can't extract `X` and `y` from the data of type {type(data)}"
                    ) from exc

        if self.shape is None:
            self.shape = args[X_name].shape[1:]

        return fn(*bound_args.args, **bound_args.kwargs)

    return wrapped_fn


def _get_xy_names(names):
    """Helper function to get names of features and targets."""
    if "X" in names:
        X_name = "X"
    elif "x" in names:
        X_name = "x"
    elif names[0] == "self":
        X_name = names[1]
    else:
        X_name = names[0]

    if "y" in names:
        y_name = "y"
    else:
        y_name = names[names.index(X_name) + 1]
    return X_name, y_name


class RecursionContext:
    """
    Raises an error if `fn` is called recursively.
    """

    def __init__(self, fn, error=None):
        self.fn = fn
        self.error = (
            RecursionError(f"{fn.__name__} doesn't allow recursion, but was called recursively.")
            if error is None
            else error
        )
        if not hasattr(fn, "_recursed"):
            fn._recursed = False

    def __enter__(self):
        if self.fn._recursed:
            raise self.error
        self.fn._recursed = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.fn._recursed = False
        return False


@default_decorator
def no_recursion(fn, error=None):
    rc = RecursionContext(fn, error)

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with rc:
            return fn(*args, **kwargs)

    return wrapped_fn


def get_defaults_from_self(fn):
    """
    Decorator that enhances a method so that if arguments are
    not passed in, then it will look for argument values in
    self.'argname'

    This functionality is invoked on arguments with default
    values None or NEED (see below; must include NEED from this
    module to use this). In the latter case, an exception is
    raised if an argument is neither passed directly nor can it
    be found at self.'argname'
    """
    sig = signature(fn)
    names = list(sig.parameters)[1:]
    defaults = {n: sig.parameters[n].default for n in names}
    defaults = {n: v for n, v in defaults.items() if v is None or v == NEED}

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        self = args[0]
        bound_args = sig.bind(*args, **kwargs)
        args = bound_args.arguments

        for k, v in defaults.items():
            if (v is None or v is NEED) and args.get(k, None) is None:
                default = getattr(self, k, None)
                if default is not None:
                    args[k] = default
                elif v is NEED:
                    raise ValueError(f"{k} must either be passed into {fn.__qualname__} or into __init__")

        return fn(*bound_args.args, **bound_args.kwargs)

    return wrapped_fn


def NEED():
    raise NotImplementedError("NEED is used only as a unique reference and should not be called.")
