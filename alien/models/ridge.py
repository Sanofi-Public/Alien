"""Wrapper for a Bayesian Ridge regression model.
Uses scikit-learn in the background.
"""

import numpy as np
from sklearn.linear_model import BayesianRidge as BayesianRidgeSKL

from ..decorators import flatten_batch
from .linear import LinearRegressor


class BayesianRidgeRegressor(LinearRegressor):
    """

    In addition to the arguments specific to this wrapper, any additional
    keyword arguments will be passed through to the constructor of
    scikit-learn's BayesianRidge class. Some important ones, quoted from
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html

    :param alpha_1: shape parameter for the Gamma distribution prior over the
        alpha parameter. default=1e-6
    :param alpha_2: inverse scale parameter (rate parameter) for the Gamma
        distribution prior over the alpha parameter. default=1e-6
    :param lambda_1: shape parameter for the Gamma distribution prior over the
        lambda parameter. default=1e-6
    :param lambda_2: inverse scale parameter (rate parameter) for the Gamma
        distribution prior over the lambda parameter. default=1e-6
    :param alpha_init: Initial value for alpha (precision of the noise). If
        not set, alpha_init is 1/Var(y).
    :param lambda_init: Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.
    """

    def __init__(self, X=None, y=None, data=None, random_seed=None, shape=None, ensemble_size=100, **kwargs):
        super().__init__(X=X, y=y, data=data, random_seed=random_seed, shape=shape, ensemble_size=ensemble_size)
        self.ensemble_size = ensemble_size
        self.model = BayesianRidgeSKL(**kwargs)

    def fit_model(self, X=None, y=None, early_stopping=None):
        self.model.fit(X, y)

    def _prepare_batch(self, X):
        return X

    def _forward(self, X, *args, **kwargs):
        return self.model.predict(X)

    @flatten_batch
    def std_dev(self, X):
        return np.sqrt((np.dot(X, self.model.sigma_) * X).sum(axis=-1))

    @property
    def weights(self):
        return self.model.coef_

    @property
    def bias(self):
        return self.model.intercept_ - np.dot(self.model.coef_, self.model.X_offset_)

    def linearization(self):
        return self.weights, self.bias

    @property
    def weight_covariance(self):
        return self.model.sigma_
