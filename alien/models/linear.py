from abc import abstractmethod

import numpy as np

from ..decorators import flatten_batch
from ..utils import ranges
from .models import CovarianceRegressor, EmbeddableModel, LastLayerEmbeddableModel


class LinearizableRegressor(CovarianceRegressor, EmbeddableModel):
    """Base class for a linearizable model."""

    def __init__(self, *args, covariance="linear", **kwargs):
        super().__init__(*args, covariance=covariance, **kwargs)

    @abstractmethod
    def linearization(self):
        """
        Finds the last-layer linearization of the model in its current
        state.

        :return: weights, bias
        """

    @flatten_batch
    def predict_linear(self, X):
        """Use the model's linearization for a prediction."""
        weights, bias = self.linearization()
        return (X * weights).sum(axis=1) + bias

    # @flatten_batch
    def covariance_linear(self, X, block_size=1000):
        if self.weight_covariance is None:
            raise RuntimeError(
                "Weight covariance hasn't been fitted since the last time the model was trained."
            )
        X = self.embedding(X)

        # We have to break up the calculation into chunks, because memory
        # usage is quite high
        cov = np.empty((*X.shape[:-2], X.shape[-2], X.shape[-2]))
        for i_0, i_1 in ranges(len(X), block_size):
            for j_0, j_1 in ranges(len(X), block_size):
                cov[..., i_0:i_1, j_0:j_1] = (
                    np.dot(X[..., i_0:i_1, None, :], self.weight_covariance)
                    * np.asarray(X)[..., None, j_0:j_1, :]
                ).sum(axis=-1)
        return cov


class LastLayerLinearizableRegressor(LinearizableRegressor, LastLayerEmbeddableModel):
    """Regressor whose last layer maps linearly to the output.
    Such models can do many nice things with the last layer
    embeddings.
    """

    def __init__(self, *args, embedding="last_layer", **kwargs):
        super().__init__(*args, embedding=embedding, **kwargs)
