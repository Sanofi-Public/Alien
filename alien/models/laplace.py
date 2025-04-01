from abc import abstractmethod

import numpy as np

from ..decorators import get_defaults_from_self, get_Xy
from ..utils import sum_except
from .linear import LinearizableRegressor
from .models import CovarianceRegressor

# pylint: disable=import-outside-toplevel

SUBCLASS_ERROR = "This method should be implemented by the subclass."


class LaplaceApproxRegressor(CovarianceRegressor):
    """
    A model which can produce last-layer embeddings (DeepChem Pytorch/Keras, vanilla
    Pytorch, or any model given an :meth:`embedding` method) may be wrapped in the
    :class:`LaplaceApproxRegressor` class, yielding an ALIEN model which computes
    covariances using the Laplace approximation on the last layer weights (see
    `Laplace Redux, Daxberger et al 2022 <https://arxiv.org/abs/2106.14806>`_)
    """

    def __init__(self, *args, lamb=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lamb = lamb
        self.weight_covariance = None

    @abstractmethod
    def fit_laplace(self, X=None, y=None):
        """
        Fits the Laplace approximation to the (last layer of)
        the model.
        """

    def fit_model(self, X=None, y=None, **kwargs):
        self.weight_covariance = None
        super().fit_model(X, y, **kwargs)

    @abstractmethod
    def covariance_laplace(self, X):
        """
        Computes covariance using the Laplace approximation
        """


class LinearizableLaplaceRegressor(LaplaceApproxRegressor, LinearizableRegressor):
    """Linearizable Laplace Regressor."""

    @get_Xy
    @get_defaults_from_self
    def fit_laplace(self, X=None, y=None, lamb=None):
        X = self.embedding(X)
        self.weight_covariance = np.linalg.inv(
            sum_except(X[..., None, :] * X[..., :, None], (-1, -2)) + lamb * np.eye(X.shape[-1])
        )

    def covariance_laplace(self, X, **kwargs):
        return self.covariance_linear(X, **kwargs)

    def _forward(self, X, *args, **kwargs):
        raise NotImplementedError(SUBCLASS_ERROR)

    def _prepare_batch(self, X):
        raise NotImplementedError(SUBCLASS_ERROR)

    def covariance(self, X, **kwargs):
        return self.covariance_laplace(X, **kwargs)

    def last_layer_embedding(self, X):
        raise NotImplementedError(SUBCLASS_ERROR)
