"""
ALIEN works with two different kinds of DeepChem models---Pytorch and Keras.
Eg., if ``dc_model`` is a DeepChem Keras model, then::

    al_model = DeepChemRegressor(dc_model)

will be an ALIEN model wrapping ``dc_model``, with covariance computed using
MC dropout. In fact, it will be an instance of :class:`DeepChemKerasRegressor`,
which you could instantiate directly (but there's no need).
"""

from functools import partial

import numpy as np

from ..config import default_training_epochs
from ..data import Dataset, DictDataset, as_DCDataset
from ..decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..utils import dict_pop, shift_seed, std_keys, update_copy, chunks
from .keras import KerasClassifier, KerasModel, KerasRegressor
from .laplace import LinearizableLaplaceRegressor
from .models import Classifier, CovarianceRegressor, Model, Output  # , WrappedModel
from .pytorch import PytorchClassifier, PytorchModel, PytorchRegressor

# pylint: disable=import-outside-toplevel


class DeepChemModel(Model):  # WrappedModel):
    """Base Deepchem wrapper."""

    def __new__(cls, model=None, X=None, y=None, mode=None, **kwargs):
        if cls == DeepChemModel:
            if not isinstance(mode, str) or mode[:7] not in {"regress", "classif"}:
                raise ValueError(
                    f"`mode` should be one of 'regress[ion/or]' or 'classifi[cation/er]', but you gave {mode}."
                )

            if mode[:7] == "regress":
                return DeepChemRegressor.__new__(DeepChemRegressor, model=model, X=X, y=y, **kwargs)
            elif mode[:7] == "classif":
                return DeepChemClassifier.__new__(DeepChemClassifier, model=model, X=X, y=y, **kwargs)
        return super().__new__(cls)

    def __init__(self, model=None, X=None, y=None, **kwargs):
        global dc  # pylint: disable=global-statement
        import deepchem as dc

        self.dc_model = model
        self.model = model.model
        super().__init__(model=model.model, X=X, y=y, **kwargs)

        fit_kwargs = dict_pop(
            kwargs,
            [
                "nb_epoch",
                "epochs",
                "epoch_limit",
            ],
            "max_checkpoints_to_keep",
            "restore",
            "loss",
            deterministic=True,
            nb_epoch=default_training_epochs,
        )
        self.fit_kwargs = fit_kwargs

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
    def fit_model(
        self,
        X=None,
        y=None,
        deterministic=True,
        val_data=None,
        **kwargs,
    ):
        if val_data is not None:
            self.val_data = val_data
        kwargs = std_keys(kwargs, ["nb_epoch", "epochs", "epoch_limit"])
        fit_kwargs = update_copy(self.fit_kwargs, deterministic=deterministic, **kwargs)
        if "nb_epoch" in fit_kwargs:
            print(f"Epochs = {fit_kwargs['nb_epoch']}")

        self.dc_model.fit(self.get_train_dataset(X, y), **fit_kwargs)

    @get_defaults_from_self
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

    def _prepare_batch(self, X, y=None):
        prepared_batch = self.dc_model._prepare_batch(([X], y, None))
        if y is None:
            return prepared_batch[0]
        return prepared_batch[:2]


class DeepChemRegressor(DeepChemModel, CovarianceRegressor):
    """The name is descriptive."""

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        import deepchem as dc

        if cls == DeepChemRegressor:
            if hasattr(dc.models, "KerasModel") and isinstance(model, dc.models.KerasModel):
                return DeepChemKerasRegressor.__new__(DeepChemKerasRegressor, model=model, X=X, y=y, **kwargs)
            if hasattr(dc.models, "TorchModel") and isinstance(model, dc.models.TorchModel):
                return DeepChemPytorchRegressor.__new__(DeepChemPytorchRegressor, model=model, X=X, y=y, **kwargs)
            raise TypeError("For DeepChem, only KerasModel or TorchModel supported.")
        return super().__new__(cls)

    def _forward(self, X, *args, iterate_inputs=None, **kwargs):
        return self.model(X, *args, **kwargs)


class DeepChemClassifier(DeepChemModel, Classifier):
    """The name is descriptive."""

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        import deepchem as dc

        if cls == DeepChemClassifier:
            if hasattr(dc.models, "KerasModel") and isinstance(model, dc.models.KerasModel):
                return DeepChemKerasClassifier.__new__(DeepChemKerasClassifier, model=model, X=X, y=y, **kwargs)
            if hasattr(dc.models, "TorchModel") and isinstance(model, dc.models.TorchModel):
                return DeepChemPytorchClassifier.__new__(DeepChemPytorchClassifier, model=model, X=X, y=y, **kwargs)
            raise TypeError("For DeepChem, only KerasModel or TorchModel supported.")
        return super().__new__(cls)

    def _forward(self, X, *args, iterate_inputs=None, **kwargs):
        # DeepChem classifiers return a tuple (prob, logit)
        if self.wrapped_output == Output.PROB:
            return self.model(X, *args, **kwargs)[0]
        if self.wrapped_output == Output.LOGIT:
            return self.model(X, *args, **kwargs)[1]
        return self.model(X, *args, **kwargs)[1].argmax(-1)
    

class DummyDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.X)
    
    def iterbatches(self, *args, **kwargs):
        return [self.data]


# -------- Pytorch models -------- #


class DeepChemPytorchModel(DeepChemModel, PytorchModel):
    def __init__(self, model=None, X=None, y=None, **kwargs):
        kwargs = std_keys(kwargs, ["epoch_limit", "nb_epoch", "epochs"])
        super().__init__(
            model=model,
            X=X,
            y=y,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, samples):
        """Collate function for Pytorch."""
        import torch

        inputs = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        inputs_b, _, _ = self.dc_model._prepare_batch(([inputs], labels, None))
        return inputs_b, torch.as_tensor(labels)

    @get_Xy
    def fit_model(self, X=None, y=None, **kwargs):
        if self.trainer in {self.dc_model, None}:
            DeepChemModel.fit_model(self, X=X, y=y, **kwargs)
        else:
            kwargs = std_keys(kwargs, ["epoch_limit", "nb_epoch", "epochs"])
            PytorchModel.fit_model(self, X=X, y=y, **kwargs)

    @flatten_batch
    def predict(self, *args, **kwargs):
        import torch

        self.model.eval()
        with torch.no_grad():
            return self.forward(*args, **kwargs)
        
    def _prepare_batch(self, X, y=None):
        import torch
        if hasattr(X, 'X'):
            if y is None:
                y = getattr(X, 'y', None)
            X = X.X

        X_p, y_p, _ = self.dc_model._prepare_batch((
            [X], [torch.tensor(0, dtype=torch.float32) if y is None else [y]], []))
        if y is None:
            return X_p
        return X_p, y_p


class DeepChemPytorchRegressor(DeepChemPytorchModel, DeepChemRegressor, PytorchRegressor):
    """The name is descriptive."""


class DeepChemPytorchClassifier(DeepChemClassifier, PytorchClassifier, DeepChemPytorchModel):
    """The name is descriptive."""

    def __init__(self, *args, output=Output.PROB, wrapped_output=Output.LOGIT, **kwargs):
        super().__init__(*args, output=output, wrapped_output=wrapped_output, **kwargs)


# -------- Keras models -------- #


class DeepChemKerasModel(DeepChemModel, KerasModel):
    """DeepChem Keras model. Inherits DeepChemModel and KerasModel."""

    def embedding(self, X):  # pylint: disable=method-hidden
        """Return embeddings from X."""
        # DeepChem has an annoying habit of padding outputs to be a multiple of the
        # batch size, and I'm not sure when it does this, hence the slicing
        return self.dc_model.predict_embedding(as_DCDataset(X))[: len(X)]

    @get_Xy
    @get_defaults_from_self
    def fit_model(self, X=None, y=None, val_data=None, **kwargs):
        self.save_initial_weights(X)

        if self.early_stopping:
            kwargs["callbacks"] = partial(
                self.early_stopping_callback,
                self.early_stopping,  # patience
                int((len(X) - 1) / (self.dc_model.batch_size)) + 1,  # epoch_length
                val_data if isinstance(val_data, Dataset) else Dataset(val_data),  # val_data
            )

        try:
            super().fit_model(X=X, y=y, **kwargs)
        except EarlyStoppingException as _:
            pass

    def early_stopping_callback(self, patience, epoch_length, val_data, model, step, verbose=False):
        epoch = int(step / epoch_length)
        if self.val_steps and self.val_steps[-1] == epoch:
            return

        self.val_steps.append(epoch)
        self.val_scores.append(self.test(val_data, metric=self.val_metric))

        # print(f"patience: {patience}, epoch_length: {epoch_length}, step: {step}, epoch: {epoch}, val_loss: {self.val_scores[-1]}")

        if verbose:
            print(f"Epoch {epoch}: val_score = {self.val_scores[-1]}")

        if self.val_scores[-1] < self.best_val_score:
            self.best_val_score = self.val_scores[-1]
            self.best_val_weights = model.model.get_weights()

        elif min(self.val_scores[-patience:]) > self.best_val_score:
            self.model.set_weights(self.best_val_weights)
            raise EarlyStoppingException(epoch - patience)

    # TODO: move to deepchem model? same implementation for both pytorch/keras

    @flatten_batch
    def predict_samples(self, X, *args, n=None, dropout_seeds=None, **kwargs):
        """Makes an ensemble of `n` predictions, using MC dropout."""
        # We have to reimplement some of DeepChem's prediction code,
        # since DeepChem's public-facing methods don't give you the ensemble
        # of predictions
        import tensorflow as tf

        X = getattr(X, 'X', X)

        n = n or self.ensemble_size
        dropout_seeds = dropout_seeds if dropout_seeds is not None else (self.rng.integers(int(1e8), size=n) if self.dropout_seeds is None else self.dropout_seeds)

        # self.training = 1
        preds = []
        for batch in chunks(X, self.dc_model.batch_size):
            inputs = self._prepare_batch(batch)
            ensemble = []

            for seed in dropout_seeds[:n]:
                tf.random.set_seed(seed)
                out = self._forward(inputs, training=1)
                if isinstance(out, list):
                    out = out[0]
                if out.ndim >= 2 and out.shape[-1] == 1:
                    out = out.numpy()[..., 0]
                ensemble.append(out)

            preds.append(tf.stack(ensemble, axis=1))
        # pylint: disable=undefined-variable
        return tf.concat(preds, axis=0)

    def _prepare_batch(self, X, y=None):
        #if y is None:
        #    breakpoint()
        import tensorflow as tf
        if hasattr(X, 'X'):
            if y is None:
                y = getattr(X, 'y', None)
            X = X.X

        ### CALL default_generator on single batch
        pre_X, pre_y, _ = next(self.dc_model.default_generator(
            DummyDataset((X, y, None, None)), 
            mode="uncertainty", 
            pad_batches=False, 
            deterministic=True
        ))

        if y is None:
            self.dc_model._create_inputs(pre_X)
            pre_y = []
        else:
            self.dc_model._create_training_ops((pre_X, pre_y, []))
        
        X_out, y_out, _ = self.dc_model._prepare_batch((pre_X, pre_y or None, None))
        if isinstance(X_out, list) and len(X_out) == 1:
            X_out = X_out[0]
        return X_out if y is None else (X_out, y_out)


class DeepChemKerasRegressor(DeepChemKerasModel, DeepChemRegressor, LinearizableLaplaceRegressor, KerasRegressor):
    """The name is descriptive."""

    def linearization(self):
        raise NotImplementedError

    def _forward(self, X, *args, iterate_inputs=None, **kwargs):
        return self.model(X, *args, **kwargs)[0]


class DeepChemKerasClassifier(DeepChemKerasModel, DeepChemClassifier, KerasClassifier):
    """The name is descriptive."""

    def reinitialize(self, *args, **kwargs):
        if hasattr(self.dc_model, "_global_step"):
            # pylint: disable=protected-access
            self.dc_model._global_step.assign(0)
        super().reinitialize(*args, **kwargs)


class EarlyStoppingException(RuntimeError):
    def __init__(self, epochs=None):
        super().__init__()
        print(f"Early stopping after {epochs} epochs.")
        self.epochs = epochs
