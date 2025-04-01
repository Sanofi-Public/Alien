"""Module for LightGBM Wrappers"""

import numpy as np

from ..decorators import flatten_batch, get_defaults_from_self
from ..tumpy import tumpy as tp
from ..utils import dict_pop, shift_seed, softmax, update_copy
from .models import EnsembleClassifier, EnsembleModel, EnsembleRegressor, Output

# pylint: disable=import-outside-toplevel


LR_DEFAULT = 0.1


class LightGBMModel:
    """
    Wrapper for LightGBM gradient boosted models, with some extra tooling
    for finding epistemic uncertainties.

    Calling this class' :meth:`.fit` method actually trains an ensemble of
    LightGBM models (of size `self.ensemble_size`). To ensure that these
    models are sampled from a 'true Bayesian posterior' (or, rather, an
    approximation of it), we modify the MSE loss function to include a
    Gaussian noise term, and we set the L2 regularization. The 'temperature'
    of the noise, as well as the strength of the L2 regularization, match
    that given in
        `Uncertainty in Gradient Boosting via Ensembles <https://arxiv.org/abs/2006.10562>`_
    (This technique was originally used in the `CatBoost <https://catboost.ai/>`_ library.)

    LightGBM parameters can be passed in either as extra keyword arguments,
    or included in the `params` dictionary argument. You can
    see the full list of parameters at
    <https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html>
    *NOTE:* If `posterior_sample` is True, then `lambda_l2` is reserved
    for use by this wrapper, and may not be included in
    `params`.

    Args:
        params: A `dict` of parameters to pass to `lightgbm.train`.
            These may instead be passed in via **kwargs.
        posterior_sample: if True, sets Langevin temperature and
            L2 regularization
        reinitialize: For a LightGBM model, `reinitialize` can be
            True, False, or one of `'refit'` or `'boost_more'` (a
            synonym for `False` in this case).
            Defaults to True.
        ensemble_size: Number of independent LightGBM models to train.
            Defaults to 10 if `posterior_sample` is True, or 1 if
            `posterior_sample` is False.
        **kwargs: Keyword arguments corresponding to LightGBM
            parameters are included as `params` to lightgbm.train.
            Other kwargs are passed to the superclass, so any arguments
            to :class:`Model` or :class:`EnsembleRegressor` will work.
    """

    def __init__(
        self,
        model=None,
        X=None,
        y=None,
        *,
        ensemble_size=None,  # defaults to 10, unless posterior_sample is False, in which case the default is 1
        ensemble=None,
        params=None,
        posterior_sample=True,
        verbose=None,
        **kwargs
    ):
        if ensemble_size is None:
            ensemble_size = 10 if posterior_sample else 1
        if params is None:
            params = {"verbosity": -1}

        super().__init__(X=X, y=y, ensemble_size=ensemble_size, **kwargs)
        import lightgbm as lgb

        self._init_ensemble(model=model, ensemble=ensemble, ensemble_size=ensemble_size)
        self.params = update_copy(params, dict_pop(kwargs, *lgbm_params_flat))

        if verbose is not None:
            self.verbose = verbose
        elif "verbosity" not in params:
            self.verbose = 0

        self.posterior_sample = posterior_sample
        if posterior_sample:
            assert (params is None) or ("lambda_l2" not in params) or (params == {}), (
                "Can't specify 'lambda_l2' in keyword args if `posterior_sample` is True. "
                "ALiEN needs to set this parameter for its own purposes."
            )

        self._learning_rate = None

        # TODO: validate for classifier and regressor
        self.temperature = 0.5

        # temperature = lambda self, l : .5 / sqrt(l * self.learning_rate)
        # temperature = lambda self, l : .5 / (l * self.learning_rate)
        # temperature = lambda self, l: 0.5
        # temperature = lambda self, l : .18

    def _init_ensemble(self, model=None, ensemble=None, ensemble_size=None):
        if ensemble is None:
            ensemble = model if type(model) in {list, tuple} else (None if model is None else [model])
        if type(ensemble) in {list, tuple}:
            self.ensemble = ensemble
            self.ensemble_size = len(ensemble)
        else:
            self.ensemble = None
            self.ensemble_size = ensemble_size

    @get_defaults_from_self
    def fit_model(self, X=None, y=None, reinitialize=None, posterior_sample=None, params=None, **kwargs):
        import lightgbm as lgb

        if params is None:
            params = {}
        params = update_copy(self.params, params)
        params.update(dict_pop(kwargs, *lgbm_params_flat))  # NOSONAR

        if posterior_sample:
            assert "lambda_l2" not in params, (
                "Can't specify 'lambda_l2' in keyword args if `posterior_sample` is True. "
                "ALiEN needs to set this parameter for its own purposes."
            )
            params.update(  # NOSONAR
                # lambda_l2 = .5 * len(X)
                lambda_l2=0.5
            )

        if not reinitialize:
            reinitialize = "boost more"

        if self.ensemble and reinitialize == "refit":
            self.ensemble = [model.refit(X, y) for model in self.ensemble]
            return

        # TODO: self.random_seed and self.noisy_loss aren't defined
        self.ensemble = [
            lgb.train(  # returns LightGBM model Booster
                update_copy(params, seed=np.random.default_rng(self.random_seed).integers(1e8)),
                lgb.Dataset(X, y),
                fobj=self.noisy_loss if posterior_sample else None,
                init_model=(self.ensemble[i] if self.ensemble and reinitialize == "boost more" else None),
                **kwargs
            )
            for i in range(self.ensemble_size)
        ]

    @property
    def learning_rate(self):
        if self._learning_rate is not None:
            return self._learning_rate
        if "learning_rate" in self.params:
            return self.params["learning_rate"]
        return LR_DEFAULT

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.params["learning_rate"] = learning_rate
        self._learning_rate = None

    @flatten_batch
    def predict_fixed_ensemble(self, X, multiple=None, **kwargs):
        assert multiple is None, "`multiple` arg not implemented yet!"
        preds = [model.predict(X, **kwargs) for model in self.ensemble]
        return tp.stack(preds, axis=1)

    @property
    def verbose(self):
        return self.params["verbosity"] + 1

    @verbose.setter
    def verbose(self, v):
        self.params["verbosity"] = v - 1


class LightGBMRegressor(LightGBMModel, EnsembleRegressor):
    """Like the name"""

    def noisy_loss(self, preds, data):
        noise = np.random.default_rng(self.random_seed).normal(scale=self.temperature, size=preds.shape)
        return preds - data.get_label() + np.array(noise), np.ones_like(preds)

    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("LightGBMRegressor does not support batching. Use fit instead.")

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError("LightGBMRegressor does not support forward method. Use predict instead.")


class LightGBMClassifier(LightGBMModel, EnsembleClassifier, EnsembleModel):
    """Like the name"""

    def __init__(self, random_seed=None, *args, **kwargs):
        super().__init__(*args, random_seed=random_seed, **kwargs)
        from lightgbm import LGBMClassifier

        self.ensemble = []
        for i in range(self.ensemble_size):
            self.ensemble.append(
                LGBMClassifier(random_state=shift_seed(self.random_seed, 17 * i), objective=self.noisy_loss)
            )

    @get_defaults_from_self
    def fit_model(self, X=None, y=None, reinitialize=None, posterior_sample=None, params=None, **kwargs):
        if params is None:
            params = {}
        for clf in self.ensemble:
            clf.fit(X, y)

    def noisy_loss(self, labels, preds):
        """Calculate noisy loss.

        Args:
            labels: np.ndarray
                The true labels. Shape (n_samples)
            preds: np.ndarray
                The predicted probabilities. Shape (n_samples, n_classes)
        """
        original_shape = preds.shape  # save original shape
        if preds.ndim == 1:
            preds = preds.reshape((len(labels), -1))

        noise = np.random.default_rng(self.random_seed).normal(
            scale=self.temperature,
            size=preds.shape,
        )

        probs = np.asarray(softmax(preds, axis=-1))

        labels = labels.astype(np.int32)
        label_probs = np.take_along_axis(probs, np.expand_dims(labels, 1), axis=-1)
        one_hots = np.eye(preds.shape[-1], dtype=probs.dtype)[labels]
        return (
            (probs - one_hots + np.array(noise)).reshape(original_shape),
            (label_probs * (one_hots - probs)).reshape(original_shape),
        )

    @flatten_batch
    def predict_fixed_ensemble(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def predict_logit(self, X, *args, **kwargs):
        # probs = self.predict_prob(X, *args, **kwargs)
        # logits = np.log(probs / probs[:, -1][:, np.newaxis])
        # logits[:, -1] = 0  # Set the logits for the baseline class (here, the last class) to 0
        # return logits

        # TODO: inverse softmax to get logits from probabilities
        # return np.stack(logits, axis=1)

        # raw_score=True  used
        preds = [model.predict_proba(X, raw_score=True, **kwargs) for model in self.ensemble]
        return tp.stack(preds, axis=1)

    def predict_prob(self, X, *args, **kwargs):
        """Predict probabilities for each class."""
        preds = [model.predict_proba(X, **kwargs) for model in self.ensemble]
        return tp.stack(preds, axis=1)

    def _prepare_batch(self, X):
        """Raises NameError."""
        raise NameError("LightGBMRegressor does not support batching. Use fit instead.")

    def _forward(self, X, *args, **kwargs):
        """Raises NameError."""
        raise NameError("LightGBMRegressor does not support forward method. Use predict instead.")


lgbm_params = [
    ["boosting", "boosting_type", "boost"],
    [
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "num_boost_round",
        "n_estimators",
        "max_iter",
    ],
    ["learning_rate"],
    ["num_leaves"],
    ["tree_learner", "tree", "tree_type", "tree_learner_type"],
    ["num_threads"],
    ["device", "device_type"],
    ["force_col_wise"],
    ["force_row_wise"],
    ["histogram pool size"],
    ["max_depth"],
    ["min_data_in_leaf", "min_data_per_leaf", "min_data", "min_samples_leaf", "min_child_samples"],
    [
        "min_sum_hessian_in_leaf",
        "min_sum_hessian_per_leaf",
        "min_sum_hessian",
        "min_hessian",
        "min_child_weight",
        "min_sum_hessian_in_leaf",
    ],
    ["bagging_fraction", "sub_row", "subsample", "bagging"],
    ["pos_bagging_fraction", "pos_sub_row", "pos_subsample", "pos_bagging"],
    ["neg_bagging_fraction", "neg_sub_row", "neg_subsample", "neg_bagging"],
    ["bagging_freq", "subsample_freq"],
    ["feature_fraction", "sub_feature"],
    ["feature_fraction_bynode", "sub_feature_bynode"],
    ["extra_tree"],
    ["max_delta_step", "max_tree_output", "max_leaf_output"],
    ["monotone_constraints", "mc", "monotone_constraint"],
    ["monotone_constraints_method", "mc_method"],
    ["monotone_penalty", "mc_penalty"],
    ["feature_contri", "feature_contrib", "fc", "fp", "feature_penalty"],
    [
        "forcedsplits_filename",
        "forced_splits_filename",
        "fs",
        "forced_splits_file",
        "forced_splits",
    ],
    ["path_smooth"],
    ["interaction_constraints"],
    ["verbosity", "verbose"],
    ["snapshot_freq"],
    ["linear_tree"],
    ["max_bin", "max_bins"],
    ["max_bin_by_feature"],
    ["min_data_in_bin"],
    ["bin_construct_sample_cnt", "subsample_for_bin"],
    ["is_enable_sparse"],
    ["enable_bundle", "bundle"],
    ["use_missing"],
    ["zero_as_missing"],
    ["feature_pre_filter"],
    ["two_round", "two_round_loading", "use_two_round_loading"],
    ["header"],
    ["label_column"],
    ["weight_column"],
    ["group_column"],
    ["ignore_column"],
    ["start_iteration_predict"],
    ["num_iteration_predict"],
    ["predict_leaf_index", "leaf_index"],
    ["predict_contrib", "contrib"],
    ["boost_from_average"],
    ["reg_sqrt"],
    ["alpha"],
    ["fair_c"],
    ["metric", "metrics", "metric_types"],
    ["gpu_platform_id"],
    ["gpu_device_id"],
    ["gpu_use_dp"],
    ["num_gpu"],
]

lgbm_params_flat = sum(lgbm_params, [])
