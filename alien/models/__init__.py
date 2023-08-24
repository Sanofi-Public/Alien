"""
This module contains wrapper classes for various kinds of ML models. To use
an externally-built model with ALIEN's selector classes, you must first wrap
it in the appropriate subclass of :class:`Model`.

The documentation for :class:`Model` explains the shared interface for all
models in the ALIEN universe.

Deep learning models
--------------------

An easy solution is to wrap your regression model (Pytorch, Keras or DeepChem)
with the :class:`Regressor` class:

.. code-block::

    wrapped_model = alien.models.Regressor(model=model, uncertainty='dropout', **kwargs)

This will tool your model to use dropouts to produce uncertainties and embeddings. Alternatively, 
you may use `uncertainty='laplace'`, in which case we will use the Laplace approximation
on the last layer of weights to produce uncertainties.

How to choose?
--------------

If you have an existing labeled dataset in a similar problem domain, you can try
running a :ref:`retrospective experiment <retrospective>` with the different
options. However, we do have some hints:

MC dropout, with the :class:`CovarianceSelector`, does best for regression
problems in our extensive benchmarks, so that's a good place to start.


Gradient boosting models
------------------------------------------

ALIEN directly supports a number of ensemble models, including

* :class:`LightGBMRegressor`
* :class:`CatBoostRegressor`

plus a number of Scikit-Learn models, listed below.

Other models
------------

ALIEN supports linear models in the form of Bayesian ridge regression
(which is convenient for getting covariances), in its Scikit-Learn
implementation:

* :class:`BayesianRidgeRegressor`

In fact, we support a number of Scikit-Learn models:

* :class:`GaussianProcessRegressor`
* :class:`RandomForestRegressor`
* :class:`ExtraTreesRegressor`
* :class:`GradientBoostingRegressor`
"""

from .deepchem import DeepChemRegressor
from .keras import KerasRegressor
from .linear import LastLayerLinearizableRegressor, LinearizableRegressor
from .models import (
    CovarianceRegressor,
    EnsembleRegressor,
    Model,
    Regressor,
    test_if_deepchem,
    test_if_keras,
    test_if_pytorch,
)

# Deprecated API
from .old_api import LaplaceApproxRegressor, MCDropoutRegressor
from .pytorch import PytorchRegressor, StdLimit, TrainingLimit, default_limit
