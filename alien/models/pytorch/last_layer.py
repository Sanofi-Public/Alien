"""Model wrapper for last layer linearization."""

# The code in this file is largely borrowed from the Laplace Redux
# implementation by Alex Immer:
#
# https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/feature_extractor.py
#
# That code is under the MIT License, which we provide a copy of here, in
# accordance with the license requirements. Note, the rest of the Active
# Learning SDK is *not* provided under the MIT license.
#
# MIT License
#
# Copyright (c) 2021 Alex Immer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from numpy.typing import ArrayLike

from ...decorators import flatten_batch
from ..models import LastLayerEmbeddingMixin

# pylint: disable=import-outside-toplevel


def is_trainable(module):
    """Return whether module has any trainable parameters."""
    return any(p.requires_grad for p in module.parameters())


class LastLayerEmbeddingPytorchMixin(LastLayerEmbeddingMixin):
    """Last layer embedding for Pytorch-based models. This is just a mixin class,
    and will not run on its own."""

    def __init__(self, model=None, *args, X=None, y=None, **kwargs):
        self.last_layer = None
        self._last_layer_name = ""
        self.has_linear_last_layer = None
        super().__init__(model=model, X=X, y=y, *args, **kwargs)

    @flatten_batch
    def predict_with_embedding(self, X, *args, **kwargs):
        """Forward pass which returns the output of the penultimate layer along
        with the output of the last layer. If the last layer is not known yet,
        it will be determined when this function is called for the first time.

        :param X: one batch of data to use as input for the forward pass
        """
        if self.last_layer is None:
            # if this is the first forward pass and last layer is unknown
            out = self.find_last_layer(X)
        else:
            # if last layer is already known
            out = self.predict(X)
        return out, self._features

    def predict_logits(self, X):
        """Forward pass which returns the logits (output of the last trainable layer)."""
        if self.last_layer is None:
            # if this is the first forward pass and last layer is unknown
            _ = self.find_last_layer(X)
        else:
            # if last layer is already known
            _ = self.predict(X)
        return self._logits

    def last_layer_embedding(self, X):
        return self.predict_with_embedding(X)[1]

    def linearization(self):
        """Return linearization of the last layer (W, b).
        Raise TypeError if last layer is not linear.
        """
        if not self.has_linear_last_layer:
            raise TypeError("Last layer on this model may not be linear.")
        return self.last_layer.weight, self.last_layer.bias

    def set_last_layer(self, last_layer_name: str) -> None:
        """Set the last layer of the model by its name. This sets the forward
        hook to get the output of the penultimate layer.

        :param last_layer_name: the name of the last layer (fixed in
            `model.named_modules()`).
        """
        import torch

        # set last_layer attributes and check if it is linear
        self._last_layer_name = last_layer_name
        self.last_layer = dict(self.model.named_modules())[last_layer_name]
        self.has_linear_last_layer = isinstance(self.last_layer, torch.nn.Linear)

        # set forward hook to extract features in future forward passes
        self.last_layer.register_forward_hook(self.hook)

    def hook(self, module, _input, output):
        self._features = _input[0].detach()
        self._logits = output.detach()

    def find_last_layer(self, X: ArrayLike):
        """Automatically determines the last layer of the model with one
        forward pass. It assumes that the last layer is the same for every
        forward pass and that it is an instance of `torch.nn.Linear`.
        Might not work with every architecture, but is tested with all PyTorch
        torchvision classification models (besides SqueezeNet, which has no
        linear last layer).

        :param X: batch of samples used to find last layer.

        :return: Returns the output of the forward pass, so as not to waste
            computation.
        """
        if self.last_layer is not None:
            raise ValueError("Last layer is already known.")

        act_out = {}

        def get_act_hook(name):
            def act_hook(_, _input, __):
                # only accepts one input
                try:
                    act_out[name] = _input[0].detach()
                except (IndexError, AttributeError):
                    act_out[name] = None
                # remove hook
                handles[name].remove()

            return act_hook

        # set hooks for all modules
        handles = {}
        for name, module in self.model.named_modules():
            handles[name] = module.register_forward_hook(get_act_hook(name))

        # check if model has more than one module
        # (there might be pathological exceptions)
        if len(handles) <= 2:
            raise ValueError("The model only has one module.")

        # forward pass to find execution order
        out = self.predict(X)

        # find the last layer, store features, return output of forward pass
        keys = list(act_out.keys())
        for key in reversed(keys):
            layer = dict(self.model.named_modules())[key]
            if len(list(layer.children())) == 0 and is_trainable(layer):
                self.set_last_layer(key)

                # save features from first forward pass
                self._features = act_out[key]

                return out

        raise ValueError("Something went wrong (all modules have children).")
