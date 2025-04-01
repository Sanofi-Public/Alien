"""Test Model base class"""

# pylint: disable=attribute-defined-outside-init

import pytest

from alien.models import Model
from alien.models.models import Output


class TestModel:
    def test_new(self):
        # This should pass initial Model.__new__ call but fail on Classifier.__new__
        with pytest.raises(TypeError):
            Model.__new__(Model, mode=None, wrapped_output=Output.PROB)
