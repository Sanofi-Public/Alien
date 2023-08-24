"""Model classes for wrapping Keras-based models into ALiEN format."""

import numpy as np

from ...decorators import flatten_batch, get_defaults_from_self, get_Xy
from ...utils import dict_pop, seed_all, shift_seed, update_copy
from ...config import default_training_epochs, INIT_SEED_INCREMENT
from ..mc_dropout import MCDropoutRegressor
from ..models import Model
from ...data import ArrayDataset

# pylint: disable=import-outside-toplevel


class KerasRegressor(MCDropoutRegressor):
    """Base Class for wrapped Keras regression models."""

    def __init__(self, model=None, X=None, y=None, **kwargs):
        self.compile_kwargs = dict_pop(
            kwargs,
            "metrics",
            "loss_weights",
            "weighted_metrics",
            "run_eagerly",
            "steps_per_execution",
            "jit_compile",
            optimizer="adam",
            loss="mse",
        )
        self.fit_kwargs = dict_pop(
            kwargs,
            "batch_size",
            "verbose",
            "callbacks",
            "validation_split",
            "validation_data",
            "shuffle",
            "class_weight",
            "sample_weight",
            "initial_epoch",
            "steps_per_epoch",
            "validation_steps",
            "validation_batch_size",
            "validation_freq",
            "max_queue_size",
            "workers",
            "use_multiprocessing",
            epochs=default_training_epochs,
        )
        global tf
        import tensorflow as tf
        self.initial_weights = None
        self.compiled = False
        self.model = model
        super().__init__(X=X, y=y, **kwargs)

    @get_defaults_from_self
    def initialize(self, init_seed=None, sample_input=None):
        if init_seed is not None:
            seed_all(init_seed)
            self.init_seed = shift_seed(init_seed, INIT_SEED_INCREMENT)

        self.save_initial_weights(sample_input)

        # Actually reinitializing the weights in keras is almost impossible,
        # so instead we shuffle the saved initial weights (within each tensor),
        # giving the same distribution
        self.set_weights(
            [
                np.random.default_rng(init_seed).permutation(w.flat).reshape(w.shape)
                for w in self.initial_weights
            ]
        )

    def get_weights(self):
        """Return model weights."""
        weights = [w.copy() for w in self.model.get_weights()]
        return None if len(weights) == 0 else weights

    @get_defaults_from_self
    def set_weights(self, initial_weights=None):
        """Set model weights according to initial_weights"""
        self.model.set_weights([w.copy() for w in initial_weights])

    def save_initial_weights(self, sample_input=None):
        """Save initial weights in self.initial_weights object."""
        if self.initial_weights is not None:
            return
        if sample_input is None:
            if self.X is None:
                raise ValueError(
                    "Can't initialize weights for the first time without sample input."
                )
            sample_input = self.X[:1]
        self.predict(sample_input)
        self.initial_weights = self.get_weights().copy()

    @get_Xy
    @get_defaults_from_self
    def fit_model(self, X=None, y=None, **kwargs):
        if not self.compiled:
            self.model.compile(**self.compile_kwargs)
            self.compiled = True
        if not (isinstance(X, tf.Tensor) or isinstance(X, np.ndarray)):
            X, y = np.asarray(X), np.asarray(y)
        self.model.fit(x=X, y=y, **update_copy(self.fit_kwargs, **kwargs))

    @flatten_batch
    def predict(self, X, **kwargs):
        #import tensorflow as tf
        if not (isinstance(X, tf.Tensor) or isinstance(X, np.ndarray)):
            X = np.asarray(X)
        preds = self.model(X, training=False, **kwargs)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds[:, 0]
        return preds

    def fix_dropouts(self):
        #import tensorflow as tf

        from .utils import modify_dropout, subobjects

        for m in subobjects(self.model, skip=self.nodropout_layers):
            if modify_dropout(m):
                self.dropouts.append(m)

    @flatten_batch
    def predict_samples(self, X, n=1, multiple=1.0):
        #import tensorflow as tf

        preds = []
        seed = self.rng.integers(1e6)

        if not (isinstance(X, tf.Tensor) or isinstance(X, np.ndarray)):
            X = np.asarray(X)

        for i in range(n):
            # get output
            tf.random.set_seed(shift_seed(seed, i * 7))
            out = self.model(X, training=1)  # training=1 sets special dropout behaviour
            if out.ndim >= 2 and out.shape[-1] == 1:
                out = out[..., 0]
            preds.append(out)

        return tf.stack(preds, axis=1)

