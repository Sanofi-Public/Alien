import copy
import warnings
from abc import ABCMeta, abstractmethod

from ..classes import abstract_group
from ..config import INIT_SEED_INCREMENT
from ..data import Dataset, DictDataset
from ..decorators import flatten_batch, get_defaults_from_self, get_Xy
from ..stats import (
    Output,
    _n_classes,
    covariance_from_ensemble,
    ensemble_from_covariance,
    entropy,
    get_output_type,
    joint_entropy,
    joint_entropy_from_covariance,
    joint_entropy_from_ensemble,
    mutual_info,
    std_dev_from_ensemble,
)
from ..tumpy import tumpy as np
from ..utils import join, ranges, shift_seed, softmax

# pylint: disable=import-outside-toplevel


# -------- Some mixin classes for embeddings -------- #


class EmbeddingMixin:
    """
    A mixin class for models which can provide a vector embedding for its
    inputs. There are a number of ways to define this embedding:

    0. A subclass defines the :meth:`embedding` method explicitly, or

    1. The model has a 'last layer' before the output, in which inputs
       are embedded, or

    2. The model's inputs are already vectors, so the embedding is the
       identity map

    Args:
        embedding (str): Specifies how this model should find the embedding.
            If it can't find an embedding method according to this guidance,
            then it raises an error.
             - `'explicit'` or `0` - Only option 0 is allowed
             - `'last_layer'` or `1` - Only option 1 is allowed
             - `'input'` or `2` - Only option 2
             - `'any'` or `[0,1,2]` - Whatever is available, preference in
               the order given
             - `'good'` or `[0,1]` - Whichever of options 0 or 1 is available,
               preference for 0. This is the default.
             - a sequence of integers - for other orderings
    """

    def __init__(self, *args, embedding="good", **kwargs):
        # breakpoint()
        super().__init__()  # *args, **kwargs)

        if hasattr(self, "embedding"):
            # Saving self.embedding so it doesn't get overwritten
            self._embedding = self.embedding

        # invokes a property setter:
        self.embedding_method = embedding

    @property
    def embedding_method(self):
        return self._embedding_method

    @embedding_method.setter
    def embedding_method(self, method):
        if not (isinstance(method, list) or isinstance(method, tuple)):
            method = [method]

        self._embedding_method = []

        for m in method:
            self._embedding_method.extend(
                {
                    "explicit": [0],
                    "last_layer": [1],
                    "input": [2],
                    "any": [0, 1, 2],
                    "good": [0, 1],
                }.get(m, [m])
            )

        self.find_method()

    method_names = {
        0: ["_embedding", "embed", "embeddings"],
        1: ["last_layer_embedding", "last_layer_embed", "embed_last_layer", "last_layer"],
        2: ["input_embedding"],
    }

    def find_method(self):
        for method in self._embedding_method:
            for a in self.method_names[method]:
                if hasattr(self, a):
                    self.embedding = getattr(self, a)
                    return

        # raise NotImplementedError(f"Could not find an embedding for model of type {type(self)}")

    def input_embedding(self, X):
        try:
            X.shape
        except AttributeError as exc:
            raise TypeError(f"`input_embedding` needs the input to be array-like, but you passed a {type(X)}") from exc
        return X

    def predict_with_embedding(self, X, *args, **kwargs):
        """
        Returns the predictions of the model on input `X`, along with the
        embedding of `X`.
        """
        warnings.warn(
            "This is an inefficient implementation of `predict_with_embedding`, \
            involving two forward passes.\
            You should override this method in your model class to avoid this inefficiency."
        )
        return self.predict(X, *args, **kwargs), self.embedding(X, *args, **kwargs)


class LastLayerEmbeddingMixin(EmbeddingMixin):
    def __init__(self, *args, embedding=None, **kwargs):
        super().__init__(*args, embedding="last_layer", **kwargs)

    @abstractmethod
    def last_layer_embedding(self, X):
        """Returns the activations of the last layer before the output."""


class Model(EmbeddingMixin, metaclass=ABCMeta):
    """
    Base class for wrapping a model. If you have a Pytorch, Keras, or DeepChem model,
    you can wrap it directly with this class. For example:

    >>> model = Model(model=your_model, mode='regression', uncertainty='dropout')

    For other model types, like LightGBM, CatBoost, and Scikit-Learn, use the subclasses
    of model specialized to those.


    Parameters
    ----------
    model
        The model to be wrapped. This can be a Pytorch, Keras, or DeepChem model. For other
        model types (eg., tree-based), use one of the specialized subclasses of `Model`.
    X, y
        You may provide training data at the time of initialization.
        You may do so by passing `X` and `y` parameters, or by passing a
        combined `data` (from which the model will extract `data.X` and
        `data.y`, if available, otherwise `data[:-1]` and `data[-1]`).

        You may instead pass in the training data when you call :meth:`.fit`.
    data
    shape
        (tuple) Specifies the `.shape` of the feature space. This will
        be set automatically if you provide training data.
    random_seed
        (int) Random seed for those models that need it. Defaults to `None`.
    init_seed
        (int) Random seed for initializing model weights. This is
        stored, and after each call to :meth:`.initialize`, it is incremented
        by `INIT_SEED_INCREMENT`.
    reinitialize_model
        (bool) Whether to reinitialize model weights before each
        :meth:`.fit`. Defaults to `True`.
    ensemble_size
        (int) Sets the ensemble size. This parameter is used by
        :meth:`.predict_fixed_ensemble` to determine how many observations to
        produce. It is also used by some ensemble models
        (eg., :class:`RandomForestRegressor` and :class:`CatBoostRegressor`)
        to set the size of their ensemble of estimators.

    early_stopping
        (int, bool) Train this many epochs with no improved validation score, but no further.
        Defaults to `False`, i.e., no early stopping.
    val_X, val_y
        (tuple) Validation data with numpy arrays instead of a data loader
    val_data
        (data loader) Validation data to use for early stopping
    val_metric
        Metric to use for early stopping. Defaults to validation loss (MSE)

    Attributes:
        data (:class:`Dataset`): The training data.
        val_data (:class:`Dataset`): The validation data.


    TODO: Document early stopping
    """

    def __new__(cls, *args, mode=None, **kwargs):
        if cls == Model:
            if mode is None and ("output" in kwargs or "wrapped_output" in kwargs):
                    mode = "classifier"

            if not isinstance(mode, str) or mode[:7] not in {"regress", "classif"}:
                raise ValueError(
                    f"`mode` should be one of 'regress[ion/or]' or 'classifi[cation/er]', but you gave {mode}."
                )

            if mode[:7] == "regress":
                return Regressor.__new__(Regressor, *args, **kwargs)
            if mode[:7] == "classif":
                return Classifier.__new__(Classifier, *args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        X=None,
        y=None,
        data=None,
        random_seed=None,
        reinitialize_model=True,
        init_seed=None,
        shape=None,
        early_stopping=False,  # run for this many epochs with no improved validation score, but no further
        val_data=None,
        val_X=None,
        val_y=None,
        val_metric=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if data is not None:
            self.data = data
            assert (X is None) and (y is None), "Only pass X,y *or* data to Model constructor"
        elif X is not None and y is None:
            self.data = X
        else:
            self._data = None
            self.X, self.y = X, y

        self.shape = shape
        self.dropouts = []

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.reinitialize_model = reinitialize_model
        self.init_seed = shift_seed(random_seed, INIT_SEED_INCREMENT) if init_seed is None else init_seed
        self.trained = False

        self.early_stopping = early_stopping
        self.val_metric = val_metric
        if val_data is not None:
            self.val_data = val_data
            assert (val_X is None) and (val_y is None), "Only pass (val_X,val_y) *or* val_data to Model constructor"
        elif X is not None and y is not None:
            self._val_data = None
            self.val_X, self.val_y = val_X, val_y

    def predict(self, X, *args, **kwargs):
        """
        Applies the model to input(s) X (with the last self.ndim
        axes corresponding to each sample), and returns prediction(s).
        """
        return self.forward(X, *args, **kwargs)

    @abstractmethod
    def _prepare_batch(self, X):
        return X

    @abstractmethod
    def _forward(self, X, *args, **kwargs):
        return self.model(X)

    def forward(self, X, *args, **kwargs):
        X = getattr(X, "X", X)
        X = self._prepare_batch(X)
        return self._forward(X, *args, **kwargs)

    @abstract_group("fit")
    @get_Xy
    @get_defaults_from_self
    def fit(
        self,
        X=None,
        y=None,
        val_X=None,
        val_y=None,
        early_stopping=None,
        reinitialize_model=None,
        fit_uncertainty=True,
        **kwargs,
    ):
        """
        Fits the model to the given training data. If `X` and `y` are
        not specified, this method looks for `self.X` and `self.y`. If
        :meth:`.fit` finds an `X` but not a `y`, it treats `X` as a
        combined dataset `data`, and then uses `X, y = data.X, data.y`.
        If we can't find `data.X` and `data.y`, we instead use
        `X, y = data[:-1], data[-1]`.

        :meth:`.fit` should also fit any accompanying uncertainty model.
        :param reinitialize: If `True`, reinitializes model weights before
            fitting. If `False`, starts training from previous weight values.
            If not specified, uses `self.reinitialize`)

        :param fit_uncertainty: If `True`, a call to :meth:`fit` will also call
            :meth:`fit_uncertainty`. Defaults to `True`.
        """
        if self._shape is None:
            self._shape = X.shape[1:]
        if reinitialize_model:
            self.reinitialize()
        if early_stopping and (reinitialize_model or not hasattr(self, "val_scores")):
            self.val_steps = []
            self.val_scores = []
            self.best_val_score = float("inf")
            self.best_val_weights = None
            self.fit_model(X=X, y=y, early_stopping=early_stopping, **kwargs)
        else:
            self.fit_model(X=X, y=y, **kwargs)
        if fit_uncertainty:
            self.fit_uncertainty(X=X, y=y)

    @abstract_group("fit")
    def fit_model(self, X=None, y=None, **kwargs):
        """
        Fit just the model component, and not the uncertainties (if these are
        computed separately)
        """

    def fit_uncertainty(self, X=None, y=None):
        """
        Fit just the uncertainties (if these need additional fitting beyond
        just the model)
        """
        if hasattr(self, "fit_laplace"):
            self.fit_laplace(X=X, y=y)

    @property
    def data(self):
        if self._data is None and self.X is not None:
            self._data = DictDataset({"X": self.X, "y": self.y})
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data, self.X, self.y = None, None, None
            return
        if not isinstance(data, Dataset):
            data = Dataset.from_data(data)
        self._data = data
        try:
            self.X, self.y = data.X, data.y
        except AttributeError:
            self.X, self.y = None, None

    @property
    def val_data(self):
        if getattr(self, "_val_data", None) and self.val_X is not None:
            self._val_data = DictDataset({"X": self.val_X, "y": self.val_y})
        return self._val_data

    @val_data.setter # NOSONAR
    def val_data(self, val_data): # NOSONAR
        if val_data is None:
            self._val_data, self.val_X, self.val_y = None, None, None
            return
        if not isinstance(val_data, Dataset):
            val_data = Dataset.from_data(val_data)
        self._val_data = val_data
        try:
            self.val_X, self.val_y = val_data.X, val_data.y
        except AttributeError:
            self.val_X, self.val_y = None, None

    @property
    def shape(self):
        """
        The shape of the feature space. Can either be specified directly,
        or inferred from training data, in which case
        `self.shape == X.shape[1:]`, i.e., the first (batch) dimension is
        dropped.

        This property is used by any methods which use the `@flatten_batch`
        decorator.
        """
        if self._shape is None and self.X is not None:
            self._shape = self.X.shape[1:]
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def ndim(self, default=1):
        """
        The number of axes in the feature space. Equal to `len(self.shape)`.
        Most commonly equal to 1. If training data have been specified, then
        `self.ndim == X.ndim - 1`.

        This property is used by any methods which use the `@flatten_batch`
        decorator.
        """
        return default if self.shape is None else len(self.shape)

    def reinitialize(self, init_seed=None, sample_input=None):
        """
        (Re)initializes the model (eg., by randomizing the weights). If `self.reinitialize_model` is True, this
        should be called at the start of every :meth:`.fit`, and this should be
        the default behaviour of :meth:`.fit`.
        """

    def save(self, path):
        """
        Saves the model. May well be overloaded by subclasses, if they contain
        non-picklable components (or pickling would be inefficient).

        For any subclass, the :meth:`.save` and :meth:`.load` methods should be
        compatible with each other.
        """
        import pickle

        # remove certain attributes/methods from the object if present
        # generator objects cant be pickled
        obj = self._prepare_save()

        # NOTE: debug purposes, trying to pickle each attribute
        # for k,v in obj.__dict__.items():
        #     try:
        #         pickle.dumps(v)
        #     except Exception as e:
        #         print(k)
        #         print(e)

        with open(path, "wb") as _file:
            pickle.dump(obj, _file)

    def _prepare_save(self):  # NOSONAR
        obj = copy.copy(self)
        obj = self._prepare_save_attrs(obj)
        obj = self._prepare_save_dropouts(obj)
        obj = self._prepare_save_hooks(obj)
        return obj

    def _prepare_save_attrs(self, obj):  # NOSONAR
        for attr in ["covariance_ensemble", "std_dev", "covariance", "last_layer"]:
            if hasattr(obj, attr):
                setattr(obj, attr, None)  # works well for methods & attrs
        return obj

    def _prepare_save_dropouts(self, obj):  # NOSONAR

        # remove certain attributes from dropouts
        if hasattr(obj, "dropouts"):
            tl_attrnames = [
                "_thread_local",
                "_metrics_lock",
                "_inbound_nodes_value",
                "_outbound_nodes_value",
            ]
            for d in obj.dropouts:
                for attr in tl_attrnames:
                    if hasattr(d, attr):
                        setattr(d, attr, None)
        return obj

    def _prepare_save_hooks(self, obj):
        if obj.model:
            if hasattr(obj.model, "_forward_hooks"):
                setattr(obj.model, "_forward_hooks", {})
        return obj

    @staticmethod
    def load(path):
        """
        Loads a model. This particular implementation only works if `.save(path)` hasn't
        been overloaded.
        """
        raise NotImplementedError("We don't load pickle files natively. Ff you would like to, use pickle.load(f)")

    def test(self, X=None, y=None, metric=None):
        raise NotImplementedError


class EntropyModel(Model):
    """
    Base class for models which can compute the entropy (information content) of
    samples, as well as the pairwise joint entropy. These properties allow this model
    to be used by `EntropySelector`.
    """

    @abstractmethod
    def entropy(self, X, **kwargs):
        """Returns the entropy (information content) of the uncertainty in the predictions
        of `X`."""

    @abstractmethod
    def joint_entropy(self, X, **kwargs):
        """Returns a matrix of the joint entropy between pairs of samples in `X`."""

    def mutual_info(self, X, **kwargs):
        """Returns the mutual information between pairs of samples in `X`."""
        entropy = self.entropy(X, **kwargs)
        joint_entropy = self.joint_entropy(X, **kwargs)
        return entropy[..., :, None] + entropy[..., None, :] - joint_entropy

    def approx_batch_entropy(self, X, **kwargs):
        """
        Approximates the joint entropy (information content) of batches of arbitrary size.
        The last batch dimension indexes the samples within each batch. Earlier batch
        dimensions index different batches.
        """
        jen = self.joint_entropy(X, **kwargs)
        n_samples = jen.shape[-1]
        i0, i1 = np.tril_indices(n_samples, -1)
        return jen[..., i0, i1].sum(axis=-1) / (n_samples - 1)

    def uncertainty(self, X, **kwargs):
        """Returns the uncertainty in the predictions of `X`."""
        return self.entropy(X, **kwargs)


class EnsembleModel(EntropyModel):
    """
    Abstract base class for models which can sample a random ensemble of predictions for each
    input. Subclasses must implement one of :meth:`predict_fixed_ensemble` or
    :meth:`predict_samples`. In exchange, they gain all the power of an `EntropyModel`.
    """

    def __init__(self, ensemble_size=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_size = ensemble_size

    @abstract_group("ensemble")
    def predict_fixed_ensemble(self, *args, **kwargs):
        """Returns an ensemble of `self.ensemble_size` predictions for each input."""
        return self.predict_samples(*args, n=self.ensemble_size, **kwargs)

    @flatten_batch
    @abstract_group("ensemble")
    def predict_samples(self, *args, n=None, ensemble_size=None, **kwargs):
        """Returns and ensemble of `n` predictions for each input.

        :param n: _description_, defaults to None
        :type n: _type_, optional
        :param ensemble_size: _description_, defaults to None
        :type ensemble_size: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        # Here, we assume `predict_fixed_ensemble` has been implemented.
        n = n or self.ensemble_size
        preds = []

        for j, k in ranges(0, n, ensemble_size):
            indices = self.rng.choice(self.ensemble_size, k - j, replace=False, shuffle=False)
            preds.append(self.predict_fixed_ensemble(*args, **kwargs)[:, indices])
        return join(preds)


class Regressor(Model):
    """
    Base class for wrapping regressors. The resulting wrapped model will compute uncertainties
    and covariances in the way prescribed by `uncertainty`.

    Args:
        model: A Pytorch, Keras or DeepChem model, to be wrapped.

        uncertainty (str): can be `'dropout'` or `'laplace'`. This determines
            how the model will compute uncertainties and covariances.

        **kwargs: You can pass in arguments to the destined subclass. So, for
            example, if `model` is a DeepChem model, then `**kwargs` may carry
            any of the arguments accepted by `alien.models.DeepChemRegressor`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        if cls == Regressor:
            if test_if_pytorch(model):
                from .pytorch import PytorchRegressor

                return PytorchRegressor.__new__(PytorchRegressor, model=model, X=X, y=y, **kwargs)
            if test_if_keras(model):
                from .keras import KerasRegressor

                return KerasRegressor.__new__(KerasRegressor, model=model, X=X, y=y, **kwargs)
            if test_if_deepchem(model):
                from .deepchem import DeepChemRegressor

                return DeepChemRegressor.__new__(DeepChemRegressor, model=model, X=X, y=y, **kwargs)
            raise TypeError(
                f"alien.models.Regressor doesn't support models of type {model.__class__.__qualname__}."
                "Perhaps there's an ALIEN model class that directly instantiates this?"
            )
        return super().__new__(cls)

    def predict(self, X, *args, **kwargs):
        """
        Applies the model to input(s) X (with the last self.ndim
        axes corresponding to each sample), and returns prediction(s).
        """
        # return self.model.predict(X, *args, **kwargs)
        return self.forward(X, *args, **kwargs)

    def test(self, X=None, y=None, metric="mse"):
        """

        Args:
            metric - 'mse'
        """
        if y is None:
            X, y = X.X, X.y

        if metric == "mse":
            return ((self.predict(X) - y) ** 2).mean()
        raise NotImplementedError("`metric` must be 'mse'.")


class CovarianceRegressor(EntropyModel, Regressor):
    def __init__(self, *args, uncertainty=None, use_covariance_for_ensemble=False, **kwargs):
        self.use_covariance_for_ensemble = use_covariance_for_ensemble
        super().__init__(*args, **kwargs)
        if uncertainty is not None:
            self.covariance = getattr(self, "covariance_" + uncertainty, self.covariance)
            self.std_dev = getattr(self, "std_dev_" + uncertainty, self.std_dev)

    def covariance(self, X):
        """
        Returns the covariance of the epistemic uncertainty between all
        rows of X. This is where memory bugs often appear, because of the
        large matrices involved.
        """
        raise NotImplementedError

    @get_defaults_from_self
    def predict_samples(self, *args, n=None, use_covariance_for_ensemble=None, **kwargs):
        """
        Returns a correlated ensemble of predictions for samples X.

        Ensembles are correlated only over the last batch dimension,
        corresponding to axis (-1 - self.
        ndim) of X. Earlier dimensions
        have no guarantee of correlation.
        """
        n = n or self.ensemble_size
        if not use_covariance_for_ensemble:
            raise RuntimeError(
                "Using covariance computation to produce ensembles, which is unusual, so we're warning you here. Set `use_covariance_for_ensemble=True` to skip this error."
            )
        mean, cov = self.predict(*args, **kwargs), self.covariance(*args, **kwargs)
        return ensemble_from_covariance(mean, cov, n, random_seed=self.rng.integers(1e8))

    # May want to override this:
    @flatten_batch
    def std_dev(self, X, **kwargs):
        """Returns the (epistemic) standard deviation of the model
        on input `X`."""
        return np.sqrt(self.covariance(X, **kwargs).diagonal())

    @flatten_batch
    def uncertainty(self, X, **kwargs):
        """Returns the (epistemic) uncertainty of the model
        on input `X`."""
        return self.std_dev(X, **kwargs)

    # ---- methods implementing the EntropyModel interface ---- #

    def entropy(self, X, **kwargs):
        """Returns the entropy (information content) of the uncertainty in the predictions
        of `X`."""
        return 2 * self.std_dev(X, **kwargs).log()

    def joint_entropy(self, X, **kwargs):
        """Returns a matrix of the joint entropy between pairs of samples in `X`."""
        cov = self.covariance(X)
        return joint_entropy_from_covariance(cov, **kwargs)


class EnsembleRegressor(EnsembleModel, CovarianceRegressor):
    """
    Inherit from EnsembleRegressor if you wish to compute ensembles directly.
    This class provides covariance, std_dev, and (mean) prediction for free, given these
    ensembles of predictions.

    Subclasses must implement one of :meth:`predict_fixed_ensemble` or
    :meth:`predict_samples`.
    """

    def __init__(self, *args, uncertainty="ensemble", **kwargs):
        super().__init__(*args, uncertainty=uncertainty, **kwargs)

    @flatten_batch
    def predict(self, *args, **kwargs):
        return self.predict_fixed_ensemble(*args, **kwargs).mean(1)

    @flatten_batch(degree=2)  # pylint: disable=no-value-for-parameter
    def covariance_ensemble(self, *args, n=None, **kwargs):
        """Compute covariance from the ensemble of predictions"""
        n = n or self.ensemble_size
        return covariance_from_ensemble(self.predict_samples(*args, n=n, **kwargs))

    @flatten_batch
    def std_dev_ensemble(self, *args, n=None, epsilon=None, rel_epsilon=1e-4, **kwargs):
        """Returns the (epistemic) standard deviation of the model
        on input `X`."""
        n = n or self.ensemble_size
        return std_dev_from_ensemble(
            self.predict_samples(*args, n=n, **kwargs), epsilon=epsilon, rel_epsilon=rel_epsilon
        )

    @flatten_batch
    def joint_entropy(self, *args, n=None, epsilon=None, rel_epsilon=1e-4, block_size=None, pbar=None, **kwargs):
        n = n or self.ensemble_size
        return joint_entropy_from_ensemble(
            self.predict_samples(*args, n=n, **kwargs),
            epsilon=epsilon,
            rel_epsilon=rel_epsilon,
            block_size=block_size,
            pbar=pbar,
        )

    @flatten_batch
    def entropy(self, *args, n=None, epsilon=None, rel_epsilon=1e-4, **kwargs):
        n = n or self.ensemble_size
        return 2 * std_dev_from_ensemble(
            self.predict_samples(*args, n=n, **kwargs),
            log=True,
            epsilon=epsilon,
            rel_epsilon=rel_epsilon,
        )

    @flatten_batch
    def uncertainty(self, X, **kwargs):
        """Returns the (epistemic) uncertainty of the model
        on input `X`."""
        return self.std_dev(X, **kwargs)


class WrappedModel(Model):
    """
    Policy for prediction methods and their names:
        predict(...) sets the model in eval mode, and torch in no_grad, then it calls forward(...)
        forward(...) preprocesses the inputs by _prepare_batch, and then sends them to _forward(...).
            Doesn't set eval/no_grad.
        _prepare_batch(...) preprocesses the inputs
        _forward(...) No preprocessing, no eval/no_grad. Takes the processed inputs, calls the
            underlying model (self.model.forward, typically), then does any post-processing before
            returning.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(model, **kwargs)

    def fit_model(self, X=None, y=None, **kwargs):
        self.model.fit(X, y)


# ---- Classifiers ---- #


class Classifier(Model):
    """
    Args:
        :param output: Output class. Specifies whether `.predict` outputs
            class IDs, probabilities, or logits.
        :param wrapped_output: ensemble of class predictions, of shape
        ... [x ensemble_size] [x n_classes]
    """

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        if cls == Classifier:
            if test_if_pytorch(model):
                from .pytorch import PytorchClassifier

                return PytorchClassifier.__new__(PytorchClassifier, model=model, X=X, y=y, **kwargs)
            if test_if_keras(model):
                from .keras import KerasClassifier

                return KerasClassifier.__new__(KerasClassifier, model=model, X=X, y=y, **kwargs)
            if test_if_deepchem(model):
                from .deepchem import DeepChemClassifier

                return DeepChemClassifier.__new__(DeepChemClassifier, model=model, X=X, y=y, **kwargs)
            raise TypeError(
                f"alien.models.Classifier doesn't support wrapping models of type {model.__class__.__qualname__}."
                "Perhaps there's an ALIEN model class that directly instantiates this?"
            )
        return super().__new__(cls)

    def __init__(
        self,
        *args,
        n_classes=None,
        output: Output = None,
        wrapped_output: Output = Output.LOGIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._n_classes = n_classes
        self.output = output or wrapped_output
        self.wrapped_output = wrapped_output

    @get_Xy
    def fit(self, X=None, y=None, *args, **kwargs):
        self._n_classes = self._n_classes or _n_classes(y)
        super().fit(X, y, *args, **kwargs)

    @property
    def n_classes(self):
        """
        The number of classes in the data. Defaults to reading from `self.y`.
        """
        if self._n_classes is None and self.y is not None:
            self._n_classes = _n_classes(self.y)
        return self._n_classes

    @n_classes.setter
    def n_classes(self, n):
        self._n_classes = n

    @property
    def wrapped_output(self):
        return self._wrapped_output

    @wrapped_output.setter
    def wrapped_output(self, output):
        self._wrapped_output = get_output_type(output)

    @get_defaults_from_self
    def predict(self, *args, output=None, **kwargs):
        output = get_output_type(output)
        # pylint: disable=undefined-variable
        if output == Output.LOGIT:
            return self.predict_logit(*args, **kwargs)
        if output == Output.PROB:
            return self.predict_prob(*args, **kwargs)
        self.predict_class(*args, **kwargs)

    def predict_class(self, *args, **kwargs):
        if self.predict.__func__ == Classifier.predict:
            raise NotImplementedError("Must override either `.predict` or `.predict_class`")
        samples = self.predict(*args, **kwargs)

        if self.wrapped_output == Output.CLASS:
            return samples
        if samples.ndim > 1 and samples.shape[-1] > 1:
            return samples.argmax(-1)  # works for numpy and torch
        else:
            return samples > (0.5 if self.wrapped_output == Output.PROB else 0)

    def predict_prob(self, *args, smoothing=0.0, **kwargs):
        if self.predict.__func__ == Classifier.predict:
            raise NotImplementedError("Must override either `.predict` or `.predict_prob`")
        if self.wrapped_output == Output.PROB:
            return self.predict(*args, **kwargs)
        if self.wrapped_output == Output.LOGIT:
            return softmax(self.predict(*args, **kwargs), axis=-1)
        if self.predict_logit.__func__ is not Classifier.predict_logit:
            return softmax(self.predict_logit(*args, **kwargs), axis=-1)
        # fallback: all answers are certain, minus smoothing
        if smoothing == 0:
            return np.eye(self.n_classes)[self.predict_class(*args, **kwargs)]
        zero_prob = smoothing / (self.n_classes - 1)
        eye = (1 - zero_prob - smoothing) * np.eye(self.n_classes) + zero_prob
        return eye(self.n_classes)[self.predict_class(*args, **kwargs)]

    def predict_logit(self, *args, smoothing=0.0, **kwargs):
        if self.predict.__func__ == Classifier.predict:
            raise NotImplementedError("Must override either `.predict` or `.predict_logit`")
        if self.wrapped_output == Output.LOGIT:
            return self.predict(*args, **kwargs)
        if self.wrapped_output == Output.PROB:
            return np.log(self.predict(*args, **kwargs))
        if self.predict_prob.__func__ is not Classifier.predict_prob:
            return np.log(self.predict_prob(*args, **kwargs))
        # fallback: all answers are certain, minus smoothing
        if smoothing == 0:
            return np.log(np.eye(self.n_classes)[self.predict_class(*args, **kwargs)])
        zero_prob = smoothing / (self.n_classes - 1)
        eye = (1 - zero_prob - smoothing) * np.eye(self.n_classes) + zero_prob
        return np.log(eye(self.n_classes)[self.predict_class(*args, **kwargs)])


class EnsembleClassifier(Classifier, EnsembleModel, EntropyModel):
    def predict_class_samples(self, *args, n=None, **kwargs):
        if n is None:
            n = self.ensemble_size
        samples = self.predict_samples(*args, n=n, **kwargs)

        if self.wrapped_output == Output.CLASS:
            return samples
        if samples.ndim > 2 and samples.shape[-1] > 1:
            return samples.argmax(-1)  # works for numpy and torch
        else:
            return samples > (0.5 if self.wrapped_output == Output.PROB else 0)

    def predict_prob_samples(self, *args, n=None, **kwargs):
        """Returns an ensemble of class probabilities, with shape [batch x n_samples x n_classes]"""
        if n is None:
            n = self.ensemble_size
        if self.wrapped_output == Output.PROB:
            return self.predict_samples(*args, n=n, *kwargs)
        if self.wrapped_output == Output.LOGIT:
            return softmax(self.predict_samples(*args, n=n, **kwargs), axis=-1)
        if self.predict_logit_samples.__func__ is not EnsembleClassifier.predict_logit_samples:
            return softmax(self.predict_logit_samples(*args, n=n, **kwargs), axis=-1)
        # fallback: all answers are certain
        return np.eye(self.n_classes)[self.predict_samples(*args, n=n, **kwargs)]

    def predict_logit_samples(self, *args, n=None, **kwargs):
        """Returns an ensemble of logits, with shape [batch x n_samples x classes]"""
        if n is None:
            n = self.ensemble_size
        if self.wrapped_output == Output.LOGIT:
            return self.predict_samples(*args, n=n, *kwargs)
        if self.wrapped_output == Output.PROB:
            return self.predict_samples(*args, n=n, **kwargs).log()
        if self.predict_prob_samples.__func__ is not EnsembleClassifier.predict_prob_samples:
            return self.predict_prob_samples(*args, n=n, **kwargs).log()
        # fallback: all answers are certain
        return np.eye(self.n_classes)[self.predict_samples(*args, n=n, **kwargs)].log()

    def predict_prob_or_class_samples(self, *args, predict_prob=None, **kwargs):
        # Helper function to get an ensemble of either class probabilites, or classes
        if self.wrapped_output == Output.CLASS or not predict_prob:
            return self.predict_class_samples(*args, **kwargs)
        return self.predict_prob_samples(*args, **kwargs)

    def entropy(self, *args, use_prob=True, **kwargs):
        """ """
        labels = Output.CLASS if self.wrapped_output == Output.CLASS or not use_prob else Output.PROB
        return entropy(
            self.predict_prob_or_class_samples(*args, predict_prob=use_prob, **kwargs),
            labels=labels,
            n_classes=self.n_classes,
        )

    def joint_entropy(self, *args, use_prob=False, pbar=False, block_size=None, **kwargs):
        """
        Returns a matrix of the pairwise joint entropy between the different samples
        """
        labels = Output.CLASS if self.wrapped_output == Output.CLASS or not use_prob else Output.PROB
        return joint_entropy(
            self.predict_prob_or_class_samples(*args, predict_prob=use_prob, **kwargs),
            n_classes=self.n_classes,
            block_size=block_size,
            pbar=pbar,
            labels=labels,
        )

    def mutual_info(self, *args, use_prob=False, **kwargs):
        """
        Returns a matrix of mutual information between the different samples.
        """
        labels = Output.CLASS if self.wrapped_output == Output.CLASS or not use_prob else Output.PROB
        return mutual_info(
            self.predict_prob_or_class_samples(*args, predict_prob=use_prob, **kwargs),
            n_classes=self.n_classes,
            labels=labels,
        )


# ---- checks to see what kind of model you have ---- #


def test_if_pytorch(model):
    pt_attrs = [
        "_parameters",
        "_buffers",
        "_forward_hooks",
        "_modules",
    ]
    if all(hasattr(model, attr) for attr in pt_attrs):
        try:
            from torch.nn import Module

            assert isinstance(model, Module)
            return True
        except ImportError:
            pass
    return False


def test_if_deepchem(model):
    # TODO: see if this works with DeepChem 2.5
    dc_attrs = [
        "_loss_fn",
        "output_types",
        "model_class",
        "_prediction_outputs",
    ]
    if all(hasattr(model, attr) for attr in dc_attrs):
        try:
            from deepchem.models import Model as DCModel

            assert isinstance(model, DCModel)
            return True
        except AssertionError:
            pass
    return False


def test_if_keras(model):
    """Test if model is a keras model."""
    kr_attrs = [
        "_supports_masking",
        "_name",
        "_callable_losses",
        "_jit_compile",
        "_input_dtype",
        "_graph_initialized",
    ]
    if sum(hasattr(model, attr) for attr in kr_attrs) >= 4:
        try:
            from tensorflow.keras import Model as KerasModel

            assert isinstance(model, KerasModel)
            return True
        except (ImportError, AssertionError):
            pass
    return False
