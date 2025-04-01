"""
This module contains wrapper classes for various kinds of ML models. To use
an externally-built model with ALIEN's selector classes, you must first wrap
it in the class :class:`Model` (or one of its subclasses)

The documentation for :class:`Model` explains the shared interface for all
models in the ALIEN universe.

Deep learning models
--------------------

An easy solution is to wrap your deep-learning model (Pytorch, Keras or DeepChem)
with the :class:`Model` class::

    wrapped_model = alien.models.Model(model=model, mode='regression', uncertainty='dropout')

`mode` can be any of `'regression'`/`'regressor'` or `'classification'`/`'classifier'`. 
Alternatively, you can wrap directly in an appropriate subclass, eg.::

    wrapped_regressor = alien.models.Regressor(model=r_model, uncertainty='dropout')
    wrapped_classifier = alien.models.Classifier(model=c_model)

The option `uncertainty='dropout'` will tool your model to use Monte Carlo dropout to produce uncertainties 
and embeddings. This is the default choice for deep learning models, and it works well *if you have
dropout layers in your architecture.*

.. warning::

    If you want to use `'dropout'` uncertainty, (empirically, the best option for differentiable models) you 
    must have dropout layers in your model. Otherwise, you will get meaningless uncertainties.

Alternatively, you may use `uncertainty='laplace'`, which will use the Laplace approximation
on the last layer of weights to produce uncertainties.

See :doc:`hyperparameters` for more info.


Gradient boosting models
------------------------------------------

ALIEN directly supports a number of popular gradient boosting models, including

* :class:`LightGBMRegressor`
* :class:`LightGBMClassifier`
* :class:`CatBoostRegressor`
* :class:`CatBoostClassifier`

plus a number of Scikit-Learn models (including gradient boosting), listed below.

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
* :class:`HistGradientBoostingRegressor`
* :class:`GaussianProcessClassifier`
* :class:`RandomForestClassifier`
* :class:`ExtraTreesClassifier`
* :class:`GradientBoostingClassifier`
* :class:`HistGradientBoostingClassifier`
"""

from .cat_boost import CatBoostClassifier, CatBoostModel, CatBoostRegressor
from .deepchem import DeepChemRegressor
from .keras import KerasClassifier, KerasRegressor
from .lightgbm import LightGBMClassifier, LightGBMRegressor
from .linear import (
    LastLayerLinearizableRegressor,
    LinearizableRegressor,
    LinearRegressor,
)
from .models import (
    Classifier,
    CovarianceRegressor,
    EnsembleClassifier,
    EnsembleRegressor,
    Model,
    Output,
    Regressor,
    test_if_deepchem,
    test_if_keras,
    test_if_pytorch,
)

# Deprecated API
from .old_api import LaplaceApproxRegressor, MCDropoutRegressor
from .pytorch import (
    PytorchClassifier,
    PytorchModel,
    PytorchRegressor,
    StdLimit,
    TrainingLimit,
    default_limit,
)
from .ridge import BayesianRidgeRegressor
from .sklearn import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
