"""Pytorch Lightning trainer."""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ...data import TupleDataset
from ...decorators import get_defaults_from_self
from ...models import Classifier, Output, Regressor
from ...utils import convert_output_type, update_copy
from .training_limits import TrainingLimit, default_limit, get_training_limit


class DefaultLightningModule(pl.LightningModule):
    """Default LightningModule for Pytorch models."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model,
        loss=F.mse_loss,
        optimizer="adam",
        al_model=None,
        wrapped_output=None,
        output=None,
        **opt_args,
    ):
        super().__init__()
        self.model = model
        self.al_model = al_model
        self.loss = loss
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.wrapped_output = wrapped_output
        self.output = output

    def forward(self, X, *args, **kwargs):
        return self.model(X, *args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), **self.opt_args)
        raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        if self.al_model is not None:
            output = self.al_model._forward(X)  # pylint: disable=protected-access
        else:
            output = self.model(X)
        if self.output is not None and self.wrapped_output is not None:
            output = convert_output_type(X, self.wrapped_output, self.output)
        
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        if self.al_model is not None:
            output = self.al_model._forward(X)  # pylint: disable=protected-access
        else:
            output = self.model(X)
        if self.output is not None and self.wrapped_output is not None:
            output = convert_output_type(X, self.wrapped_output, self.output)
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss


def get_dataloader(X=None, y=None, batch_size=None, collate_fn=None):
    if y is None:
        if isinstance(X, torch.utils.data.DataLoader):
            return X
        if isinstance(X, torch.utils.data.Dataset):
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn)
        if isinstance(X, TupleDataset) and len(X.data) == 2:
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn)
        # includes cases where X is a Dataset or a tensor/array of some sort
        return torch.utils.data.DataLoader(
            TupleDataset((X[..., :-1], X[..., -1])),
            collate_fn=collate_fn,
        )
    return torch.utils.data.DataLoader(TupleDataset((X, y)), batch_size=batch_size, collate_fn=collate_fn)


class LightningTrainer:
    """Pytorch Lightning trainer."""

    @get_training_limit
    def __init__(
        self,
        model,
        training_limit: Optional[TrainingLimit] = default_limit,
        batch_size=None,
        collate_fn=None,
        deterministic=True,
        al_model=None,
        loss=None,
        output=None,
        **kwargs,
    ):
        if al_model is not None and hasattr(al_model, "wrapped_output"):
            self.wrapped_output = al_model.wrapped_output
        else:
            self.wrapped_output = None
        self.model = self._init_model(model, al_model=al_model, loss=loss, output=output)

        self.training_limit = training_limit
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        kwargs["deterministic"] = deterministic
        self.al_model = al_model
        self.kwargs = kwargs
        self.loss = loss

    def _init_model(self, model, al_model=None, loss=None, output=None):
        if isinstance(model, pl.LightningModule):
            model_ = model
        elif isinstance(model, torch.nn.Module):
            if loss is None:
                if isinstance(al_model, Classifier):
                    loss = F.cross_entropy
                elif isinstance(al_model, Regressor):
                    loss = F.mse_loss
            if loss is F.cross_entropy:
                output = Output.LOGIT
            if al_model is not None and hasattr(al_model, "wrapped_output"):
                wrapped_output = al_model.wrapped_output
            else:
                wrapped_output = None
            model_ = DefaultLightningModule(
                model, al_model=al_model, loss=loss, wrapped_output=wrapped_output, output=output
            )
        return model_

    @get_training_limit
    @get_defaults_from_self
    def fit(
        self,
        X,
        y,
        batch_size=64,
        val_loader=None,
        collate_fn=None,
        training_limit=None,
        **kwargs,
    ):
        train_loader = get_dataloader(X, y, batch_size=batch_size, collate_fn=collate_fn)
        training_limit = training_limit if training_limit is not None else self.training_limit
        steps = training_limit.batch_limit(batch_size=train_loader.batch_size, length=len(train_loader.dataset))

        new_kwargs = update_copy(self.kwargs, min_steps=steps, max_steps=steps, **kwargs)
        pl.Trainer(**new_kwargs).fit(self.model, train_loader, val_loader)
