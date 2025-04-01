"""
This module contains wrapped versions of a variety of scikit-learn models. With a few
exceptions, these models behave like the scikit-learn models they wrap.

Some notably missing models:
- Gaussian process classifiers; scikit-learn's implementation does not support sampling,
    or other means of getting joint entropies.
"""

import warnings

import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.gaussian_process as gaussian_process
from joblib import Parallel, delayed

from ..decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..stats import joint_entropy
from ..utils import dict_pop, ranges, shift_seed, version_number
from .models import Classifier, EnsembleModel, EnsembleRegressor, Model, Output

VERSION = version_number(sklearn)
if VERSION < (1, 3):
    # pylint: disable=import-error,no-name-in-module
    from sklearn.ensemble import BaseGradientBoosting
    from sklearn.ensemble._gb_losses import LeastSquaresError, MultinomialDeviance
else:
    # pylint: disable=protected-access
    from sklearn._loss import (  # AbsoluteError,; ExponentialLoss,; HuberLoss,; PinballLoss,
        HalfMultinomialLoss,
        HalfSquaredError,
    )
    from sklearn.ensemble._gb import BaseGradientBoosting

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        learning_rate=0.1,
        X_csc=None,
        X_csr=None,
    ):
        raw_predictions *= 1 - learning_rate * self._loss.regularization
        return BaseGradientBoosting._fit_stage(
            self=self,
            i=i,
            X=X,
            y=y,
            raw_predictions=raw_predictions,
            sample_weight=sample_weight,
            sample_mask=sample_mask,
            random_state=random_state,
            X_csc=X_csc,
            X_csr=X_csr,
        )

    sklearn.ensemble.GradientBoostingRegressor._fit_stage = _fit_stage
    sklearn.ensemble.GradientBoostingClassifier._fit_stage = _fit_stage


sklearn_models = [
    cls
    for cls in sklearn.ensemble.__dict__.values()
    if isinstance(cls, type)
    and (issubclass(cls, sklearn.base.RegressorMixin) or issubclass(cls, sklearn.base.ClassifierMixin))
]


def al_kwargs(kwargs):
    """
    Pop out kwargs that are not supported by sklearn
    """
    kwargs = kwargs.copy()
    dict_pop(
        kwargs,
        "X",
        "y",
        "data",
        "shape",
        "uncertainty",
        "random_seed",
        "random_state",
        "output",
        "loss",
        "max_iter",
        "ensemble_size",
    )
    return kwargs


class LangevinMixin:
    """
    Mixin class for loss functions for both regressors and classifiers
    """

    def __init__(self, sample_weight=None, n_classes=None, temperature=0.5, regularization=0.5, random_seed=None):
        # NOTE: dirty fix regressors vs. classifiers
        if n_classes is not None:  # classifiers
            super().__init__(sample_weight=sample_weight, n_classes=n_classes)
        else:  # regression
            super().__init__(sample_weight=sample_weight)
        self.temperature = temperature if VERSION < (1, 4) else temperature / 2
        self.rng = np.random.default_rng(random_seed)
        self.regularization = regularization

    ##### sklearn v < 1.3 methods

    def negative_gradient(self, y, raw_predictions, **kwargs):
        ng_res = super().negative_gradient(y, raw_predictions)
        return ng_res + self.rng.normal(scale=self.temperature, size=ng_res.shape)

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        raw_predictions *= 1 - learning_rate * self.regularization
        return super().update_terminal_regions(
            tree,
            X,
            y,
            residual,
            raw_predictions,
            sample_weight,
            sample_mask,
            learning_rate=0.1,
            k=k,
        )

    ##### sklearn v1.3+ methods

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        gradient_out = super().gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        gradient_out += self.rng.normal(scale=self.temperature, size=gradient_out.shape)
        return gradient_out

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        loss_out, gradient_out = super().loss_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        gradient_out += self.rng.normal(scale=self.temperature, size=gradient_out.shape)
        return loss_out, gradient_out

    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        proba_out=None,
        n_threads=1,
    ):
        gradient_out, proba_out = super().gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            proba_out=proba_out,
            n_threads=n_threads,
        )
        gradient_out += self.rng.normal(scale=self.temperature, size=gradient_out.shape)
        return gradient_out, proba_out

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        gradient_out, hessian_out = super().gradient_hessian(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            hessian_out=hessian_out,
            n_threads=n_threads,
        )
        gradient_out += self.rng.normal(scale=self.temperature, size=gradient_out.shape)
        return gradient_out, hessian_out


if VERSION < (1, 3):

    class LangevinLeastSquares(LangevinMixin, LeastSquaresError):
        pass

    class LangevinMultinomial(LangevinMixin, MultinomialDeviance):
        pass

else:

    class LangevinLeastSquares(LangevinMixin, HalfSquaredError):
        pass

    class LangevinMultinomial(LangevinMixin, HalfMultinomialLoss):
        pass


# Register the new loss function with sklearn
# skgblosses.LOSS_FUNCTIONS['langevin'] = LangevinLoss
# skgb.GradientBoostingRegressor._parameter_constraints['loss'][0].options.add('langevin')
# print("\n\n!!!!!!!! DEAL WITH LINE 230 SKLEARN.PY REGISTER LOSS !!!!!!!!\n\n")


def _check_params_dummy():
    pass


class SKLearnModel(Model):
    """Base class for all Scikit-learn wrappers."""

    def __init__(
        self,
        *args,
        model=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sklearn_kwargs = {}
        if "ensemble_size" in kwargs:
            self.sklearn_kwargs["n_estimators"] = kwargs.pop("ensemble_size")
        kwargs.update(self.sklearn_kwargs)

        for cls in sklearn_models:
            if isinstance(model, cls):
                break
            if model == cls or (isinstance(model, str) and model.lower() in cls.__name__.lower()):
                #  initialize single model instance with defined size of the ensemble
                model = cls(random_state=self.rng.integers(1e8), **al_kwargs(kwargs))
                break

        self.model = model

    def fit_model(self, X=None, y=None, *args, **kwargs):
        self.model.fit(X, y)

    def covariance(self, X):
        raise NameError("Covariance not implemented for SKLearnModel.")


class SKLearnEnsembleModel(SKLearnModel, EnsembleModel):
    """
    Wraps scikit-learn ensemble models (classifier / regressors).
    Uncertainties and covariances are computed from the statistics of the ensemble
    of predictions.

    You can pass in all the usual arguments to an :class:`EnsembleRegressor` or
    :class:`EnsembleClassifier`, as well as any arguments to the scikit-learn class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = []
        self.not_implemented_error = "This method is not implemented for SKLearnEnsembleModel."

    def get_predict_fn(self):
        "Get prediction function for each estimator"
        return "predict"

    @flatten_batch
    @get_defaults_from_self
    def predict_fixed_ensemble(self, X, *args, **kwargs):
        """Returns an ensemble of predictions."""
        return self.predict_samples(X, n=self.ensemble_size, *args, **kwargs)

    @flatten_batch
    @get_defaults_from_self
    def predict_samples(self, X, n: int = None, *args, **kwargs):
        preds = []
        n = n or self.ensemble_size

        if n < self.ensemble_size:
            for model in self.get_rand_models(n):
                preds.append(self.call_predict_fn(model, X))
        else:  # reuse the whole ensemble more than once
            for j, k in ranges(0, n, self.ensemble_size):
                for model in self.get_rand_models(k - j):
                    preds.append(self.call_predict_fn(model, X))

        res_tensor = np.stack(preds, 1)
        # add last dimension if tensor has < 3 dims (single elements in last dim)
        res_tensor = np.expand_dims(res_tensor, -1) if res_tensor.ndim < 3 else res_tensor

        return res_tensor

    def call_predict_fn(self, model, X):
        predict_fn = getattr(model, self.get_predict_fn())  # get the specific function to call
        return np.asarray(predict_fn(X))

    def get_rand_models(self, n):
        # either estimators_ or self.models
        models = self.model.estimators_ if self.model else self.models
        # select subset of models based on `n`
        indices = self.rng.choice(self.ensemble_size, n, replace=False, shuffle=False)

        return [models[i] for i in indices]

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError()

    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("RandomForestRegressor does not support batching. Use fit instead.")

    def entropy(self, X, **kwargs):
        """Raises NameError."""
        raise NameError(self.not_implemented_error)

    def joint_entropy(self, X, **kwargs):
        """Raises NameError."""
        raise NameError(self.not_implemented_error)

    def test(self, X=None, y=None, metric=None):
        raise NameError(self.not_implemented_error)


class SKLearnEnsembleClassifier(SKLearnEnsembleModel, Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_fns = {
            Output.LOGIT: "predict_log_proba",
            Output.PROB: "predict_proba",
            Output.CLASS: "predict",
        }

    def get_predict_fn(self):
        return self.predict_fns[self.output]

    @flatten_batch
    @get_defaults_from_self
    def predict_samples(self, X, n: int = 1, *args, **kwargs):
        """
        No need to pass self.ensemble_size here as the models have been pre-trained already
        """

        preds = []

        if n < self.ensemble_size:
            for model in self.get_rand_models(n):
                preds.append(self.call_predict_fn(model, X))
        else:  # reuse the whole ensemble more than once
            for j, k in ranges(0, n, self.ensemble_size):
                for model in self.get_rand_models(k - j):
                    preds.append(self.call_predict_fn(model, X))

        res_tensor = np.stack(preds, 1)
        # add last dimension if tensor has < 3 dims (single elements in last dim)
        res_tensor = np.expand_dims(res_tensor, -1) if res_tensor.ndim < 3 else res_tensor

        return res_tensor

    def call_predict_fn(self, model, X):
        predict_fn = getattr(model, self.get_predict_fn())  # get the specific function to call
        return np.asarray(predict_fn(X))

    @flatten_batch
    def predict(self, X, *args, **kwargs):
        return self.predict_fixed_ensemble(X, *args, **kwargs).mean(1)

    def get_rand_models(self, n):
        # TODO: this should return something
        models = self.model.estimators_ if self.model is not None else self.models
        # select subset of models based on `n`
        indices = np.random.default_rng(self.random_seed).choice(self.ensemble_size, n, replace=False, shuffle=False)
        return [models[i] for i in indices]

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError("This method is not implemented for RandomForestClassifier.")

    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("RandomForestClassifier does not support batching. Use fit instead.")

    def entropy(self, X, **kwargs):
        """Raises NameError."""
        raise NameError("This method is not implemented for RandomForestClassifier.")

    def predict_prob_or_class_samples(self, *args, predict_prob=None, **kwargs):
        # Helper function to get an ensemble of either class probabilites, or classes
        if self.wrapped_output == Output.CLASS or not predict_prob:
            return self.predict_class(*args, **kwargs)
        return self.predict_prob(*args, **kwargs)

    def joint_entropy(self, *args, use_prob=False, pbar=False, block_size=None, **kwargs):
        """
        Returns a matrix of the pairwise joint entropy between the different samples
        """
        return joint_entropy(
            self.predict_prob_or_class_samples(*args, predict_prob=use_prob, **kwargs),
            n_classes=self.n_classes,
            block_size=block_size,
            pbar=pbar,
        )


class RandomForestRegressor(SKLearnEnsembleModel):
    """
    A random forest regressor, based on the scikit-learn implementation.
    Uncertainties and covariances are computed from the predictions of the ensemble of trees.

    You can pass in all the usual arguments to an :class:`EnsembleRegressor`,
    as well as any arguments to the scikit-learn class.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model="RandomForestRegressor", loss=LangevinLeastSquares, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.predict_fixed_ensemble(X, *args, **kwargs).mean(1)


class RandomForestClassifier(SKLearnEnsembleClassifier):
    """
    A random forest classifier, based on the scikit-learn implementation.
    Entropies and information content are computed from the predictions of the
    ensemble of trees.

    You can pass in all the usual arguments to an :class:`EnsembleRegressor`,
    as well as any arguments to the scikit-learn class.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model="RandomForestClassifier", loss=LangevinMultinomial, **kwargs)


class ExtraTreesRegressor(SKLearnEnsembleModel):
    """
    An extra-trees (extremely random forest) regressor, based on the scikit-learn
    implementation. Uncertainties and covariances are computed from the predictions of
    the ensemble of trees.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model="ExtraTreesRegressor", loss=LangevinLeastSquares, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.predict_fixed_ensemble(X, *args, **kwargs).mean(1)


class ExtraTreesClassifier(SKLearnEnsembleClassifier):
    """
    An extra-trees (extremely random forest) classifier, based on the scikit-learn
    implementation. Entropies and information content are computed from the predictions of the
    ensemble of trees.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model="ExtraTreesClassifier", loss=LangevinMultinomial, **kwargs)


class GradientBoosting(SKLearnModel):
    def __init__(
        self,
        cls,
        loss,
        n_jobs=-1,
        X=None,
        y=None,
        *args,
        n_classes=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_jobs = n_jobs
        self.loss = loss
        self.n_classes = n_classes
        self.models = []
        self.cls = cls
        self.sklearn_kwargs.update(kwargs)
        if "ensemble_size" in self.sklearn_kwargs:
            self.sklearn_kwargs["n_estimators"] = self.sklearn_kwargs.pop("ensemble_size")
        for key in {"output", "wrapped_output"}:
            if key in self.sklearn_kwargs:
                self.sklearn_kwargs.pop(key)
        self.X, self.y = X, y

    def build_models(self):
        # pylint: disable=protected-access
        self.models = []
        for i in range(self.ensemble_size):
            # initialization of the inner model class
            m = self.cls(random_state=shift_seed(self.random_seed, 17 * i), **self.sklearn_kwargs)
            # m._check_params()
            m._check_params = _check_params_dummy
            m._get_loss = self.loss

            # pass n_classes parameter for classifiers
            if self.loss in [LangevinMultinomial]:
                m._loss = self.loss(n_classes=self.n_classes)
            else:
                m._loss = self.loss()
            if self.cls == sklearn.ensemble.HistGradientBoostingClassifier:
                m.loss = m._loss
            self.models.append(m)

    def fit_model(self, X=None, y=None, **kwargs):
        # NOTE: debugging purposes without parallelization
        # for m in self.models:
        #    m.fit(X, y, **kwargs)
        self.X = X
        self.y = y
        if not self.models:
            self.build_models()

        # parallelized version
        self.models = Parallel(n_jobs=self.n_jobs)(delayed(m.fit)(X, y, **kwargs) for m in self.models)


class GradientBoostingRegressor(GradientBoosting, SKLearnEnsembleModel, EnsembleRegressor):
    """
    This class wraps the GradientBoostingRegressor class of sklearn.ensemble. As with other gradient
    boosting models within ALIEN, this actually fits an ensemble of models, using Langevin dynamics to
    (approximately) sample these models from the true posterior.

    Uncertainties and covariances are computed from the predictions of this ensemble of models.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_iter: The maximum number of iterations to run the gradient boosting algorithm for.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # pass specific cls to the constructor
        super().__init__(
            cls=sklearn.ensemble.GradientBoostingRegressor,
            loss=LangevinLeastSquares,
            *args,
            **kwargs,
        )


class GradientBoostingClassifier(GradientBoosting, SKLearnEnsembleClassifier):
    """
    This class wraps the GradientBoostingClassifier class of sklearn.ensemble. As with other gradient
    boosting models within ALIEN, this actually fits an ensemble of models, using Langevin dynamics to
    (approximately) sample these models from the true posterior.

    Entropies and information content are computed from the predictions of the ensemble of models.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_iter: The maximum number of iterations to run the gradient boosting algorithm for.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # pass specific cls to the constructor
        super().__init__(
            cls=sklearn.ensemble.GradientBoostingClassifier,
            loss=LangevinMultinomial,
            *args,
            **kwargs,
        )


class HistGradientBoostingRegressor(GradientBoosting, SKLearnEnsembleModel, EnsembleRegressor):
    """
    This class wraps the HistGradientBoostingRegressor class of sklearn.ensemble. As with other gradient
    boosting models within ALIEN, this actually fits an ensemble of models, using Langevin dynamics to
    (approximately) sample these models from the true posterior.

    Uncertainties and covariances are computed from the predictions of this ensemble of models.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_iter: The maximum number of iterations to run the gradient boosting algorithm for.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            cls=sklearn.ensemble.HistGradientBoostingRegressor,
            loss=LangevinLeastSquares,
            *args,
            **kwargs,
            # **std_keys(kwargs, ["max_iter", "n_estimators"]),
        )
        if "n_estimators" in self.sklearn_kwargs:
            self.sklearn_kwargs.pop("n_estimators")

    # def build_models(self, cls, **kwargs):
    # super().build_models(cls, **std_keys(kwargs, ["max_iter", "n_estimators"]))


class HistGradientBoostingClassifier(GradientBoosting, SKLearnEnsembleClassifier):
    """
    This class wraps the HistGradientBoostingClassifier class of sklearn.ensemble. As with other gradient
    boosting models within ALIEN, this actually fits an ensemble of models, using Langevin dynamics to
    (approximately) sample these models from the true posterior.

    Entropies and information content are computed from the predictions of the ensemble of models.

    Args:
        X, y, data, shape, random_seed, init_seed, reinitialize_model: see :class:`Model`
        ensemble_size, n_estimators (either): The number of trees in the ensemble.
        max_iter: The maximum number of iterations to run the gradient boosting algorithm for.
        max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose,
        warm_start, ccp_alpha, max_samples, monotonic_cst: See the `SKLearn docs <link URL>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            cls=sklearn.ensemble.HistGradientBoostingClassifier,
            loss=LangevinMultinomial,
            *args,
            **kwargs,
            # **std_keys(kwargs, ["max_iter", "n_estimators"]),
        )
        if "n_estimators" in self.sklearn_kwargs:
            self.sklearn_kwargs.pop("n_estimators")

        self.predict_fns = {
            Output.LOGIT: "predict_log_proba",
            Output.PROB: "predict_proba",
            Output.CLASS: "predict",
        }

    # sklearn.ensemble.HistGradientBoostingClassifier is missing `predict_log_proba` method
    def predict_log_proba(self, *args, **kwargs):
        raise NotImplementedError("Predicting logits is not implemented for GaussianProcessClassifier")


class GaussianProcessModel(SKLearnEnsembleModel):
    def __init__(
        self,
        kernel="Matern",
        shape=None,
        random_seed=None,
        X=None,
        y=None,
        max_fit_attempts=100,
        *args,
        **kwargs,
    ):
        super().__init__(shape=shape, random_seed=random_seed, X=X, y=y, *args, **kwargs)

        if kernel == "Matern":
            kernel = gaussian_process.kernels.Matern
        elif kernel == "RBF":
            kernel = gaussian_process.kernels.RBF
        if issubclass(kernel, gaussian_process.kernels.Kernel):
            kernel = kernel(length_scale_bounds=(1e-8, 1e8))

        assert isinstance(kernel, gaussian_process.kernels.Kernel), (
            "kernel argument must be either a string 'Matern', 'RBF', or "
            "a subclass of sklearn.gaussian_process.kernels.Kernel, or an "
            "instance of such a class."
        )

        self.X_scale = 1
        self.max_fit_attempts = max_fit_attempts
        self.kernel = kernel

    @get_Xy
    def fit_model(self, X=None, y=None, **kwargs):
        """
        See models.Model.fit for details.
        In this case, X and y are automatically normalized
        before fitting (and this is reversed during prediction).
        So, no need to normalize before passing to this method.
        """
        if self.shape is not None:
            assert X.shape[1:] == self.shape, "X is of the wrong shape."

        # gaussian process requires a 2d array, even if fitting on 1d
        X = np.array(X)
        if X.ndim == 1:
            X = X[:, None]

        X_ptp = np.ptp(X, axis=0)
        X_ptp[X_ptp == 0] = 1
        self.X_scale = 1.0 / X_ptp
        X = X * self.X_scale
        # NOTE: variance in y is internally normalized to 1 within the GaussianProcessRegressor

        # gaussian process fitting often raises convergence errors,
        # and this usually indicates a bad fit, so in such cases
        # we should repeat the fitting
        # pylint: disable=import-outside-toplevel
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            for _ in range(self.max_fit_attempts - 1):
                try:
                    # The actual fitting:
                    self.model.fit(X, y)
                except ConvergenceWarning:
                    pass
                else:
                    self.trained = True
                    break
        if not self.trained:
            # try one more time
            self.model.fit(X, y)

    def make_2d(self, X):
        # gaussian process requires a 2d array, even if fitting on 1d
        X = np.asarray(X)
        if X.ndim == 1:
            if self.ndim == 0:
                X = X[:, None]
            else:
                X = X[None, :]
        return X

    @flatten_batch
    def predict_fixed_ensemble(self, X):
        return self.predict_samples(X, n=1)


class GaussianProcessRegressor(GaussianProcessModel):
    """
    Wraps the GaussianProcessRegressor class of sklearn.gaussian_process.

    :param kernel: can either be one of {'Matern', 'RBF'}, or a subclass
        of gaussian_process.kernels.Kernel, or an actual
        instance of a kernel. In the first two cases, **kwargs
        are passed to the kernel constructor.
    :param X: Optionally, you can provide a reference to a training
        dataset, which will be stored, and called upon during
        fitting.
    :param y: reference to labels in training dataset. If X is given but
        y is not, then the inputs are X[:, :-1] and the labels
        are y = X[: , -1]
    :param random_seed: A random state initializer passed to GaussianProcessModel,
        defaults to None
    :param max_fit_attempts: If the GaussianProcessModel's fit function raises
        a convergence error, the fitting will be repeated. This
        parameters limits how many attempts it will make.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=self.random_seed,
        )

    @flatten_batch
    def predict(self, X):
        return self.model.predict(self.make_2d(X) * self.X_scale, return_std=False)

    @flatten_batch
    def predict_samples(self, X, n=None):
        return self.model.sample_y(np.asarray(X) * self.X_scale, n_samples=n or self.ensemble_size)

    @flatten_batch
    def std_dev(self, X):
        return self.model.predict(self.make_2d(X) * self.X_scale, return_std=True)[1]

    @flatten_batch
    def covariance(self, X):
        return self.model.predict(self.make_2d(X) * self.X_scale, return_cov=True)[1]

    def test(self, X=None, y=None, metric=None):
        raise NotImplementedError("Test method is not implemented for GaussianProcessRegressor.")


class GaussianProcessClassifier(GaussianProcessModel, SKLearnEnsembleClassifier):
    "Not yet implemented."

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        # self.model = gaussian_process.GaussianProcessClassifier(
        #     kernel=self.kernel,
        #     n_restarts_optimizer=3,
        #     random_state=self.random_seed,
        # )

        # self.predict_fns = {
        #     Output.LOGIT: "predict_log_proba",
        #     Output.PROB: "predict_proba",
        #     Output.CLASS: "predict",
        # }
        raise NotImplementedError("GaussianProcessClassifier is not yet implemented.")

    # gaussian_process.GaussianProcessClassifier is missing `predict_log_proba` method
    def predict_log_proba(self, *args, **kwargs):
        raise NotImplementedError("Predicting logits is not implemented for GaussianProcessClassifier")

    @flatten_batch
    def predict(self, X):
        predict_fn = getattr(self.model, self.get_predict_fn())
        return predict_fn(X)

    @flatten_batch
    def predict_samples(self, X, n=None):
        # TODO: we should be working with the ensemble instead
        # there is no equivalent of sample_y method in GaussianProcessClassifier
        # return self.model.sample_y(np.asarray(X) * self.X_scale, n_samples=n)
        # return self.predict(X)
        raise NotImplementedError(
            "`predict_samples` method is not implemented for GaussianProcessClassifier, see notes."
        )
