alien.models
====================

.. automodule:: alien.models
   :members:

Base classes
---------------------

The superclass for all model wrapper classes:

.. autoclass:: alien.models.Model
   :members:

The classes you will instantiate to wrap your deep learning models:

.. autoclass:: alien.models.Regressor
   :members:

.. autoclass:: alien.models.Classifier
   :members:

Then we have several abstract base classes, defining the class hierarchy:

.. autoclass:: alien.models.CovarianceRegressor

.. autoclass:: alien.models.EnsembleRegressor

Other models
------------

We have some special classes for wrapping popular models from Scikit-learn,
CatBoost, and LightGBM.

.. autoclass:: alien.models.LightGBMRegressor

.. autoclass:: alien.models.CatBoostRegressor

.. autoclass:: alien.models.GaussianProcess

.. autoclass:: alien.models.RandomForestRegressor

.. autoclass:: alien.models.BayesianRidgeRegressor


