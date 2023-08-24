from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from ...decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..laplace import LinearizableLaplaceRegressor
from ..mc_dropout import MCDropoutRegressor
from ...config import INIT_SEED_INCREMENT
from .last_layer import LastLayerPytorchLinearization
from .training_limits import default_limit
from .utils import as_tensor, dropout_forward, submodules, pl_argnames
from ...utils import dict_pop, shift_seed

# imports of torch occur inside __init__ to avoid import when not used
# pylint: disable=import-outside-toplevel


def init_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        import torch

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 2:
                torch.nn.init.kaiming_uniform_(
                    module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif module.weight.dim() == 1:
                bound = sqrt(6 / module.weight.shape[0])
                torch.nn.init.uniform_(module.weight, -bound, bound)
            else:
                torch.nn.init.uniform_(module.weight, -sqrt(2), sqrt(2))
        init_bias(module, torch)


def init_bias(module, torch):
    if hasattr(module, "bias") and module.bias is not None:
        if module.weight.dim() >= 2:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        else:
            bound = sqrt(2)
        torch.nn.init.uniform_(module.bias, -bound, bound)


class PytorchRegressor(LastLayerPytorchLinearization, MCDropoutRegressor, LinearizableLaplaceRegressor):
    """
    :param trainer: Specifies how the model will be trained.
        May be:
            'model' --- calls self.model.fit
            'lightning' --- trains with pytorch-lightning
            trainer --- calls trainer.fit
            None --- chooses from the above in order, if available
    """

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
        **kwargs,
    ):
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

        pl_kwargs = dict_pop(kwargs, *pl_argnames)

        super().__init__(
            X=X,
            y=y,
            random_seed=random_seed,
            **kwargs
        )

        # if no trainer is provided, choose one based on what's available
        if trainer is None:
            if hasattr(model, "fit"):
                trainer = "model"
            else:
                try:
                    import pytorch_lightning

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
                **pl_kwargs
            )
        else:
            self.trainer = trainer

    def fix_dropouts(self):
        import torch
        for name, module in submodules(self.model, skip=self.nodropout_layers):
            if isinstance(module, torch.nn.Dropout):
                module.forward = dropout_forward.__get__(module)
                self.dropouts.append(module)

    def get_lightning_trainer(self, random_seed=None, **kwargs):
        """Rerturn a LightningTrainer object from current model."""
        from .lightning import LightningTrainer

        if random_seed is not None:
            from pytorch_lightning import seed_everything

            seed_everything(shift_seed(random_seed, 31523), workers=True)
            kwargs["deterministic"] = True
        return LightningTrainer(self.model, **kwargs)

    @get_defaults_from_self
    def fit_model(self, X=None, y=None, reinitialize=None, init_seed=None, **kwargs):
        self.trainer.fit(X, y, **kwargs)
    

    @get_defaults_from_self
    def initialize(self, init_seed=None, sample_input=None):
        import torch

        if init_seed is not None:
            torch.manual_seed(init_seed)
            self.init_seed = shift_seed(init_seed, INIT_SEED_INCREMENT)
        self.model.apply(init_weights)

    @flatten_batch
    def predict(self, X, return_std_dev=False, convert_dtype=True):
        import torch

        try:
            X = as_tensor(X)
            if convert_dtype:
                X = X.type(self.dtype)
        except (ValueError, TypeError, RuntimeError):
            pass

        with torch.no_grad():
            self.model.eval()
            return self.model(X)

    @flatten_batch
    def predict_samples(self, X, n=1, multiple=1.0, convert_dtype=True):
        assert multiple == 1

        self.model.eval()
        for d in self.dropouts:
            d.training = 1

        try:
            X = as_tensor(X)
            if convert_dtype:
                X = X.type(self.dtype)
        except (ValueError, TypeError, RuntimeError):
            pass

        with torch.no_grad():
            return torch.stack([self.model(X).squeeze() for _ in range(n)], dim=1)

    @property
    def dtype(self):
        if getattr(self, '_dtype', None) is None:
            self._dtype = next(iter(self.model.parameters())).dtype
        return self._dtype

