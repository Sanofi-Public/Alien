"""
ALIEN works with two different kinds of DeepChem models---Pytorch and Keras.
Eg., if ``dc_model`` is a DeepChem Keras model, then::

    al_model = DeepChemRegressor(dc_model)

will be an ALIEN model wrapping ``dc_model``, with covariance computed using
MC dropout. In fact, it will be an instance of :class:`DeepChemKerasRegressor`,
which you could instantiate directly.
"""

import numpy as np

from ..data import DictDataset, as_DCDataset
from ..decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..utils import dict_pop, update_copy, as_numpy, std_keys, shift_seed
from ..config import default_training_epochs
from .keras import KerasRegressor
from .models import CovarianceRegressor
from .pytorch import PytorchRegressor
from .laplace import LinearizableLaplaceRegressor

# pylint: disable=import-outside-toplevel


class DeepChemRegressor(CovarianceRegressor):
    """Base Deepchem regressor wrapper."""

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        import deepchem as dc

        if cls == DeepChemRegressor:
            if hasattr(dc.models, "KerasModel") and isinstance(model, dc.models.KerasModel):
                return DeepChemKerasRegressor.__new__(
                    DeepChemKerasRegressor, model=model, X=X, y=y, **kwargs
                )
            elif hasattr(dc.models, "TorchModel") and isinstance(model, dc.models.TorchModel):
                return DeepChemPytorchRegressor.__new__(
                    DeepChemPytorchRegressor, model=model, X=X, y=y, **kwargs
                )
            else:
                raise TypeError("For DeepChem, only KerasModel or TorchModel supported.")
        else:
            return super().__new__(cls)

    def __init__(self, model=None, X=None, y=None, **kwargs):
        global dc
        import deepchem as dc

        fit_kwargs = dict_pop(
            kwargs,
            ["nb_epoch", "epochs", "epoch_limit",],
            "max_checkpoints_to_keep",
            "restore",
            "loss",
            deterministic=True,
            nb_epoch=default_training_epochs,
        )
        super().__init__(model=model.model, X=X, y=y, **kwargs)
        self.fit_kwargs = fit_kwargs
        self.dc_model = model
        self.model = model.model

    def get_train_dataset(self, X=None, y=None, **kwargs):
        """Get DeepChem dataset from X and y.

        Args:
            X (_type_, optional): _description_. Defaults to None.
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # pylint: disable=undefined-variable
        if isinstance(X, dc.data.Dataset):
            return X
        if isinstance(X, DictDataset):
            return X._to_DC()
        return dc.data.NumpyDataset(X=X, y=y, **kwargs)

    @get_defaults_from_self
    def fit_model(self, X=None, y=None, reinitialize=None, deterministic=True, init_seed=None, **kwargs):
        kwargs = std_keys(kwargs, ['nb_epoch', 'epochs', 'epoch_limit'])
        fit_kwargs = update_copy(self.fit_kwargs, deterministic=deterministic, **kwargs)
        if "nb_epoch" in fit_kwargs:
            print(f"Epochs = {fit_kwargs['nb_epoch']}")

        self.dc_model.fit(self.get_train_dataset(X, y), **fit_kwargs)

    def fit(self, X=None, y=None, **kwargs):
        """
        Trains this model on the given dataset, using the
        `deepchem.models.Model.fit` method.
        `X` and `y` follow the usual rules of the `fit` method,
        and may be any data-amenable type, including DeepChem Datasets.

        **kwargs will be passed along to DeepChem, and these are very
        important to the training, so we reproduce the standard .fit
        keyword arguments here:

        Parameters
        ----------
        nb_epoch: int
            the number of epochs to train for
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        deterministic: bool
            if True, the samples are processed in order.  If False, a different random
            order is used for each epoch.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        """
        super().fit(X=X, y=y, **kwargs)

    @flatten_batch
    def predict(self, X):
        # DeepChem has an annoying habit of padding outputs to be a multiple of the
        # batch size, and I'm not sure when it does this, hence the slicing
        out = self.dc_model.predict(as_DCDataset(X))[: len(X)]
        if out.ndim >= 2 and out.shape[-1] == 1:
            out = out[..., 0]
        return out


class DeepChemPytorchRegressor(DeepChemRegressor, PytorchRegressor):
    def __init__(self, model=None, X=None, y=None, **kwargs):
        kwargs = std_keys(kwargs, ['epoch_limit', 'nb_epoch', 'epochs'])
        super().__init__(
            model=model,
            X=X,
            y=y,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, samples):
        import torch

        inputs = [[s[0] for s in samples]]
        labels = torch.tensor([s[1] for s in samples], dtype=torch.float32)
        return self.dc_model._prepare_batch((inputs, labels, None))[0], labels

    @get_Xy
    def fit_model(self, X=None, y=None, **kwargs):
        if self.trainer in {self.dc_model, None}:
            DeepChemRegressor.fit_model(self, X=X, y=y, **kwargs)
        else:
            kwargs = std_keys(kwargs, ['epoch_limit', 'nb_epoch', 'epochs'])
            PytorchRegressor.fit_model(self, X=X, y=y, **kwargs)

    @flatten_batch
    def ___predict(self, X):
        import torch

        with torch.no_grad():
            self.model.eval()
            return self.model(self.dc_model._prepare_batch(([X], None, None))[0])

    @flatten_batch
    def predict_samples(self, X, n=1, multiple=1.0):
        return super().predict_samples(self.dc_model._prepare_batch(([X], None, None))[0], n=n)


class DeepChemKerasRegressor(DeepChemRegressor, KerasRegressor, LinearizableLaplaceRegressor): 

    def embedding(self, X):
        # DeepChem has an annoying habit of padding outputs to be a multiple of the
        # batch size, and I'm not sure when it does this, hence the slicing
        return self.dc_model.predict_embedding(as_DCDataset(X))[: len(X)]

    def linearization(self):
        raise NotImplementedError

    @get_Xy
    def fit_model(self, X=None, y=None, **kwargs):
        self.save_initial_weights(X)
        super().fit_model(X=X, y=y, **kwargs)

    @flatten_batch
    def predict_samples(self, X, n=1, multiple=1.0):
        """Makes an ensemble of `n` predictions, using MC dropout.
        """
        # We have to reimplement some of DeepChem's prediction code,
        # since DeepChem's public-facing methods don't give you the ensemble
        # of predictions
        import tensorflow as tf

        X = as_DCDataset(X)
        preds = np.empty((len(X), n))
        seed = self.rng.integers(1e6)

        for n in range(n):
            generator = self.dc_model.default_generator(
                X, mode="uncertainty", pad_batches=False, deterministic=True
            )

            batch_pred = []
            for batch in generator:
                # prepare batch
                inputs, labels, weights = batch
                self.dc_model._create_inputs(inputs)
                inputs, _, _ = self.dc_model._prepare_batch((inputs, None, None))
                if len(inputs) == 1:
                    inputs = inputs[0]

                # get output
                tf.random.set_seed(shift_seed(seed, n * 7))
                out = self.model(inputs, training=1)  # dropout=True)#, training=True)
                if isinstance(out, list):
                    out = out[0]
                if out.ndim >= 2 and out.shape[-1] == 1:
                    out = out.numpy()[..., 0]
                batch_pred.append(out)
            preds[:, n] = np.concatenate(batch_pred, axis=0)[: len(X)]
        return preds

