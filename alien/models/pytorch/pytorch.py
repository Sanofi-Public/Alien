from math import sqrt

from ...config import INIT_SEED_INCREMENT
from ...decorators import flatten_batch, get_defaults_from_self
import numpy as np
#from ...tumpy import tumpy as np
from ...utils import dict_pop, shift_seed
from ..laplace import LinearizableLaplaceRegressor
from ..mc_dropout import MCDropoutClassifier, MCDropoutModel, MCDropoutRegressor
from ..models import Classifier, Regressor  # , WrappedModel
from .last_layer import LastLayerEmbeddingPytorchMixin
from .training_limits import default_limit
from .utils import (
    as_tensor,
    dropout__getstate__,
    dropout_forward,
    pl_argnames,
    submodules,
)

# imports of torch occur inside __init__ to avoid import when not used
# pylint: disable=import-outside-toplevel


def init_weights(module):
    """Initialize weights and biases of a module."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        import torch

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 2:
                torch.nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif module.weight.dim() == 1:
                bound = sqrt(6 / module.weight.shape[0])
                torch.nn.init.uniform_(module.weight, -bound, bound)
            else:
                torch.nn.init.uniform_(module.weight, -sqrt(2), sqrt(2))
        init_bias(module, torch)


def init_bias(module, torch):
    """Initialize bias of a module."""
    if hasattr(module, "bias") and module.bias is not None:
        if module.weight.dim() >= 2:
            # pylint: disable=protected-access
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        else:
            bound = sqrt(2)
        torch.nn.init.uniform_(module.bias, -bound, bound)


class PytorchModel(LastLayerEmbeddingPytorchMixin, MCDropoutModel):  # , WrappedModel
    """

    Args:
        trainer: Specifies how the model will be trained.
            May be:
                'model' --- calls self.model.fit
                'lightning' --- trains with pytorch-lightning
                trainer --- calls trainer.fit
                None --- chooses from the above in order, if available

        mode: Specifies the type of model. May be:
            'regression'/'regressor' --- for regression models
            'classification'/'classifier' --- for classification models
    """

    def __new__(cls, *args, mode=None, **kwargs):
        if cls is PytorchModel:
            if not isinstance(mode, str) or mode[:7] not in {"regress", "classif"}:
                raise ValueError(
                    f"`mode` should be one of 'regress[ion/or]' or 'classifi[cation/er]', but you gave {mode}."
                )

            if mode[:7] == "regress":
                return PytorchRegressor.__new__(PytorchRegressor, *args, **kwargs)
            if mode[:7] == "classif":
                return PytorchClassifier.__new__(PytorchClassifier, *args, **kwargs)
        return super().__new__(cls, *args, mode=mode, **kwargs)

    def __init__(
        self,
        model=None,
        X=None,
        y=None,
        trainer=None,
        batch_size=64,
        training_limit=default_limit,
        collate_fn=None,
        random_seed=None,
        convert_dtype=True,
        iterate_inputs=False,
        stack_outputs=False,
        stack_samples="inner",  # or 'outer'
        call_predict=None,
        loss=None,
        **kwargs,
    ):
        """
        A few quick notes about some parameters:

        Args:

            iterate_inputs: If `True`, during prediction iterates through the
                rows and calls the model on each one individually, returning
                the results as a new instance of the same class as the inputs
                (commonly, either a list or an `ObjectDataset`.)
            stack_outputs: If stack_outputs is `True`, attempts to 'stack' the
                outputs (either as a Tensor or ndarray) before returning.
            stack_samples: When `predict_samples` (or `predict_fixed_ensemble`) is called
                and the output is a list or ObjectDataset or has `dtype==object`,
                this setting determines how the samples will be stacked.
                `'outer'` means the outer array will be a `bdim + 1` (typically, 2)-D array.
                `'inner'` means the outer array will not change shape, but stacking
                will happen within each row.
        """
        # imports occur inside __init__ to avoid import when not used:
        global torch  # pylint: disable=global-statement
        import torch

        assert (
            isinstance(model, torch.nn.Module) or model is None
        ), f"model is of type {type(model)}. Should be torch.nn.Module or None."
        self.model = model
        self.batch_size = batch_size
        self.training_limit = training_limit
        self.collate_fn = collate_fn
        self._dtype = None

        pl_kwargs = dict_pop(kwargs, *pl_argnames)

        super().__init__(model=model, X=X, y=y, random_seed=random_seed, **kwargs)

        self.convert_dtype = convert_dtype
        self.iterate_inputs = iterate_inputs
        self.stack_outputs = stack_outputs
        assert stack_samples in {"outer", "inner", "none", None}
        self.stack_samples = stack_samples

        # if no trainer is provided, choose one based on what's available
        if trainer is None:
            if hasattr(model, "fit"):
                trainer = "model"
            else:
                try:
                    import pytorch_lightning  # pylint: disable=unused-import

                    trainer = "lightning"
                except ImportError:
                    pass

        if trainer == "model":
            self.trainer = model
        elif trainer == "lightning":
            self.trainer = self.get_lightning_trainer(
                random_seed=random_seed,
                collate_fn=collate_fn,
                training_limit=training_limit,
                loss=loss,
                batch_size=self.batch_size,
                **pl_kwargs,
            )
        else:
            self.trainer = trainer

    def fix_dropouts(self):
        # pylint: disable=no-value-for-parameter
        super().fix_dropouts()

        import torch

        self.dropouts = []
        for _, module in submodules(self.model, skip=self.nodropout_layers):
            if isinstance(module, torch.nn.Dropout):
                module.forward = dropout_forward.__get__(module)
                module.__getstate__ = dropout__getstate__.__get__(module)
                self.dropouts.append(module)

    def get_lightning_trainer(self, random_seed=None, loss=None, **kwargs):
        """Return a LightningTrainer object from current model."""
        import torch.nn.functional as F

        from .lightning import LightningTrainer

        if loss is None:
            if isinstance(self, Classifier):
                loss = F.cross_entropy
            elif isinstance(self, Regressor):
                loss = F.mse_loss

        if random_seed is not None:
            from pytorch_lightning import seed_everything

            seed_everything(shift_seed(random_seed, 31523), workers=True)
            kwargs["deterministic"] = True
        return LightningTrainer(self.model, al_model=self, **kwargs)

    @get_defaults_from_self
    def fit_model(self, X=None, y=None, reinitialize=None, init_seed=None, batch_size=None, **kwargs):
        self.trainer.fit(X, y, batch_size=batch_size, **kwargs)

    @get_defaults_from_self
    def reinitialize(self, init_seed=None, sample_input=None):
        import torch

        if init_seed is not None:
            torch.manual_seed(init_seed)
            self.init_seed = shift_seed(init_seed, INIT_SEED_INCREMENT)
        self.model.apply(init_weights)

    def predict(self, *args, **kwargs):
        import torch

        self.model.eval()
        with torch.no_grad():
            return self.forward(*args, iterate_inputs=self.iterate_inputs, **kwargs)

    def _forward(self, X, iterate_inputs=None, *args, **kwargs):
        if iterate_inputs:
            import torch

            return (torch.stack if self.stack_outputs else X.__class__)(
                [self.model.forward(u, *args, **kwargs) for u in X]
            )
        return self.model.forward(X, *args, **kwargs)

    def _prepare_batch(self, X, *args, **kwargs):
        import torch

        if isinstance(X, torch.Tensor):
            return X if not self.convert_dtype else X.type(self.dtype, non_blocking=True)
        elif isinstance(X, np.ndarray):
            return torch.as_tensor(X) if not self.convert_dtype else torch.as_tensor(X).type(self.dtype, non_blocking=True)
        elif isinstance(X, tuple):
            return tuple(self._prepare_batch(u) for u in X)
        elif isinstance(X, list):
            return [self._prepare_batch(u) for u in X]
        elif isinstance(X, dict):
            return {k: self._prepare_batch(v) for k, v in X.items()}
        
        try:
            X = torch.as_tensor(X)
            if self.convert_dtype:
                X = X.type(self.dtype)
        except (ValueError, TypeError, RuntimeError):
            pass
        return X

    @flatten_batch
    def predict_samples(self, X, *args, n=None, seeds=None, multiple=1.0, **kwargs):
        for key in {"block_size", "pbar", "use_prob"}:
            if key in kwargs:
                kwargs.pop(key)
        n = n or self.ensemble_size
        self.model.eval()
        for dropout in self.dropouts:
            dropout.training = 1  # set to 1 to check later on in "dropout_forward" method

        X = self._prepare_batch(X)
        return self._predict_step(X, *args, n=n, dropout_seeds=seeds, **kwargs)

    def _predict_step(self, X, *args, n=None, dropout_seeds=None, **kwargs):
        # pylint: disable=undefined-variable
        dropout_seeds = (
            dropout_seeds
            if dropout_seeds is not None
            else (self.rng.integers(int(1e8), size=n) if self.dropout_seeds is None else self.dropout_seeds)
        )
        if len(dropout_seeds) < n:
            raise ValueError(f"You're asking for {n} samples, but you provided only {len(dropout_seeds)} seeds/")

        with torch.no_grad():
            samples = []
            for seed in dropout_seeds[:n]:
                self.seed_dropouts(seed)

                # TODO: fix iterate_inputs parameter error while testing
                samples.append(self._forward(X, *args, iterate_inputs=self.iterate_inputs, **kwargs))

            if isinstance(samples[0], list):
                if isinstance(samples[0][0], torch.Tensor):
                    samples = [torch.stack(u) for u in samples]
                else:
                    samples = [np.asarray(u) for u in samples]

            if getattr(samples[0], "dtype", None) == object:
                if self.stack_samples == "outer":
                    return np.stack(samples, 1)
                if self.stack_samples == "inner":
                    return [torch.stack([s[i] for s in samples]) for i in range(len(X))]
                return [[s[i] for s in samples] for i in range(len(X))]

            return torch.stack(samples, 1) if isinstance(samples[0], torch.Tensor) else np.stack(samples, 1)

    def seed_dropouts(self, seed):
        for dropout in self.dropouts:
            dropout.seed = seed
            seed += 57

    @property
    def dtype(self):
        if getattr(self, "_dtype", None) is None:
            self._dtype = next(iter(self.model.parameters())).dtype
        return self._dtype


class PytorchRegressor(PytorchModel, MCDropoutRegressor, LinearizableLaplaceRegressor):
    """Like the name"""


class PytorchClassifier(PytorchModel, MCDropoutClassifier):
    """Like the name"""
