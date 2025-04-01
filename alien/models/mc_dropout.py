"""Wrapper for different models that implement Monte Carlo dropout."""
import numpy as np

from abc import abstractmethod

from ..utils import as_list
from .models import EnsembleClassifier, EnsembleModel, EnsembleRegressor
from .utils import get_base_model

# Used to be: MCDropoutRegressor


# pylint: disable=import-outside-toplevel
class MCDropoutModel(EnsembleModel):
    """
    Wraps a deep-learning model, giving it the ability to estimate
    uncertainties and covariances by using dropout during inference
    to produce an ensemble of predictions. (See
    `Gal & Ghahramani 2016 <https://arxiv.org/abs/1506.02142>`_.)

    *NOTE:* If your model has frozen layers, i.e., if it has a pretrained
    part which won't be trained any further, and if those layers have
    dropout, then it's important for you to let `MCDropoutModel` know!
    Frozen layers should not use dropout during inference, since for
    active learning we're not interested in model uncertainty for these
    layers (since they won't benefit from the new labelled data).
    See the parameters `frozen_layers` and `nodropout_layers`.

    :param model: The model to wrap.
        Must be an instance of one of
            - :class:`torch.nn.Module`
            - :class:`tensorflow.keras.Model`
            - :class:`deepchem.models.Model`

    :param nodropout_layers: (Synonym for `frozen_layers`) The part of the
        model that should *not* have dropout turned on during inference.
        `nodropout_layers` can be a submodule of `model` or a collection
        of submodules. Default is the empty set.

        In a typical use case, these layers are an initial part of the model
        which has been pretrained on unlabeled data, with a downstream model
        trained on labeled data. Since the goal of active learning is to
        select new data for labeled training, we don't want these
        pretrained layers to influence the computed model uncertainty. In
        other words, we don't care how much the pretrained model might
        change if it had more data, since we aren't selecting data for
        pretraining.

    :param frozen_layers: Synonym for `nodropout_layers`
    """

    def __init__(
        self,
        model=None,
        X=None,
        y=None,
        ensemble_size=400,  # No extra training cost, so we bump it up a bit
        uncertainty="dropout",
        nodropout_layers=frozenset(),
        frozen_layers=None,
        dropout_seeds = None,
        **kwargs
    ):
        if uncertainty == "dropout":
            uncertainty = "ensemble"
        if model is not None:
            self.model = model
        super().__init__(X=X, y=y, uncertainty=uncertainty, ensemble_size=ensemble_size, **kwargs)

        if dropout_seeds == 'fixed':
            self.dropout_seeds = self.rng.integers(int(1e8), size=ensemble_size)
        elif isinstance(dropout_seeds, int):
            self.dropout_seeds = np.default_rng(dropout_seeds).integers(int(1e8), size=ensemble_size)
        else:
            self.dropout_seeds = dropout_seeds

        if frozen_layers is not None:
            nodropout_layers = frozen_layers
        self.nodropout_layers = {get_base_model(m) for m in as_list(nodropout_layers)}
        self.fix_dropouts()

    @abstractmethod
    def fix_dropouts(self):
        """
        Retools dropouts for MC dropout prediction

        Inheritor is responsible for calling `super().fix_dropouts()` *or*
        for setting self.dropouts = []
        """
        self.dropouts = []  # This should appear in any implementor's fix_dropouts

    def __setstate__(self, state):
        """unpickled objects will have unmodified dropouts"""
        self.__dict__.update(state)
        self.fix_dropouts()


class MCDropoutRegressor(MCDropoutModel, EnsembleRegressor):
    """This class just mixes MCDropoutModel and EnsembleRegressor."""


class MCDropoutClassifier(MCDropoutModel, EnsembleClassifier):
    """This class just mixes MCDropoutModel and EnsembleClassifier"""
