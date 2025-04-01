"""
We define some functions and classes which replicate the old API, with deprecation warnings
"""
from deprecated import deprecated

from . import Regressor


# pylint: disable=abstract-class-instantiated
@deprecated(version=1.1, reason="Use `Regressor` class with uncertainty='dropout' argument.")
def MCDropoutRegressor(model, X=None, y=None, **kwargs):
    return Regressor(model=model, X=X, y=y, uncertainty="dropout", **kwargs)


@deprecated(version=1.1, reason="Use `Regressor` class with uncertainty='laplace' argument.")
def LaplaceApproxRegressor(model, X=None, y=None, **kwargs):
    return Regressor(model=model, X=X, y=y, uncertainty="laplace", **kwargs)
