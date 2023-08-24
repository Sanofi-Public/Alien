from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ...data import TupleDataset
from ...decorators import get_defaults_from_self
from ...utils import update_copy
from .training_limits import TrainingLimit, default_limit, get_training_limit


class DefaultLightningModule(pl.LightningModule):
    # pylint: disable=unused-argument
    def __init__(self, model, loss=F.mse_loss, optimizer="adam", **opt_args):
        super().__init__()
        self.model = model
        self.loss = loss

        self.optimizer = optimizer
        self.opt_args = opt_args

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), **self.opt_args)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        output = self.model(X)
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        output = self.model(X)
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss


def get_dataloader(X=None, y=None, batch_size=None, collate_fn=None):
    if y is None:
        if isinstance(X, torch.utils.data.DataLoader):
            return X
        elif isinstance(X, torch.utils.data.Dataset):
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn)
        elif isinstance(X, TupleDataset) and len(X.data) == 2:
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn)
        else:  # includes cases where X is a Dataset or a tensor/array of some sort
            return torch.utils.data.DataLoader(
                TupleDataset((X[..., :-1], X[..., -1])),
                collate_fn=collate_fn,
            )
    else:
        return torch.utils.data.DataLoader(
            TupleDataset((X, y)), batch_size=batch_size, collate_fn=collate_fn
        )


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
        **kwargs,
    ):
        if isinstance(model, pl.LightningModule):
            self.model = model
        elif isinstance(model, torch.nn.Module):
            self.model = DefaultLightningModule(model)

        self.training_limit = training_limit
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        kwargs["deterministic"] = deterministic
        self.kwargs = kwargs

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
        training_limit = training_limit if training_limit is not None \
            else self.training_limit
        steps = training_limit.batch_limit(
            batch_size=train_loader.batch_size, length=len(train_loader.dataset)
        )

        new_kwargs = update_copy(self.kwargs, min_steps=steps, max_steps=steps, **kwargs)
        pl.Trainer(**new_kwargs).fit(self.model, train_loader, val_loader)
