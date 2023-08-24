from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ..data import Dataset, DictDataset
from ..decorators import flatten_batch, abstract_group, get_Xy, get_defaults_from_self
from ..stats import covariance_from_ensemble, ensemble_from_covariance, std_dev_from_ensemble
from ..utils import shift_seed, ranges, join
from ..config import INIT_SEED_INCREMENT

# pylint: disable=import-outside-toplevel


class Model(metaclass=ABCMeta):
    """
    Abstract base class for wrapping a model.
    Implementers must provide prediction and
    fitting (training) methods.


    Parameters
    ----------
    X
        You may provide training data at the time of initialization.
        You may do so by passing `X` and `y` parameters, or by passing a
        combined `data` (from which the model will extract `data.X` and
        `data.y`, if available, otherwise `data[:-1]` and `data[-1]`).

        You may instead pass in the training data when you call :meth:`.fit`.
    y
    data
    shape
        Specifies the `.shape` of the feature space. This will
        be set automatically if you provide training data.
    random_seed
        Random seed for those models that need it.
    init_seed
        Random seed for initializing model weights. This is
        stored, and after each call to :meth:`.initialize`, it is incremented
        by `INIT_SEED_INCREMENT`.
    reinitialize
        Whether to reinitialize model weights before each
        :meth:`.fit`. Defaults to `True`.
    ensemble_size
        Sets the ensemble size. This parameter is used by
        :meth:`.predict_ensemble` to determine how many observations to
        produce. It is also used by some ensemble models
        (eg., :class:`RandomForestRegressor` and :class:`CatBoostRegressor`)
        to set the size of their ensemble of estimators.
    """

    def __init__(
        self,
        X=None,
        y=None,
        data=None,
        random_seed=None,
        reinitialize=True,
        init_seed=None,
        shape=None,
        ensemble_size=40,
        **kwargs
    ):
        super().__init__()

        if data is not None:
            self.data = data
            assert (X is None) and (y is None), "Only pass X,y *or* data to Model constructor"
        elif X is not None and y is None:
            self.data = X
        else:
            self._data = None
            self.X, self.y = X, y

        self.shape = shape

        self.ensemble_size = ensemble_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.reinitialize = reinitialize
        self.init_seed = (
            shift_seed(random_seed, INIT_SEED_INCREMENT) if init_seed is None else init_seed
        )
        self.trained = False

    @abstractmethod
    def predict(self, X):
        """
        Applies the model to input(s) X (with the last self.ndim
        axes corresponding to each sample), and returns prediction(s).
        """

    def predict_samples(self, X, n=1):
        """
        Makes a prediction for for the batch X, randomly selected from
        this model's posterior distribution. Gives an
        ensemble of predictions, with shape `(len(X), n)`.
        """
        return join(self.predict_samples(X) for _ in range(n))

    @abstract_group('fit')
    @get_Xy
    @get_defaults_from_self
    def fit(self, X=None, y=None, reinitialize=None, fit_uncertainty=True, **kwargs):
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
        if reinitialize:
            self.initialize()
        self.fit_model(X=X, y=y, **kwargs)
        if fit_uncertainty:
            self.fit_uncertainty(X=X, y=y)

    @abstract_group('fit')
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
        if hasattr(self, 'fit_laplace'):
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

    def initialize(self, init_seed=None, sample_input=None):
        """
        (Re)initializes the model weights. If `self.reinitialize` is True, this 
        should be called at the start of every :meth:`.fit`, and this should be 
        the default behaviour of :meth:`.fit`.
        """
        pass

    def save(self, path):
        """
        Saves the model. May well be overloaded by subclasses, if they contain
        non-picklable components (or pickling would be inefficient).

        For any subclass, the :meth:`.save` and :meth:`.load` methods should be
        compatible with each other.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Loads a model. This particular implementation only works if `.save(path)` hasn't
        been overloaded.
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


class Regressor(Model):
    """
    This class can accept as its first argument (or `model`),
    any of the deep learning models we currently support. So,
    Pytorch, Keras or DeepChem.

    `Regressor`'s constructor will build a specialized subclass depending on
    the type of `model`. The resulting wrapped model will compute uncertainties 
    and covariances in the way prescribed by `uncertainty`.

    Args:
        model: A Pytorch, Keras or DeepChem model, to be wrapped.

        uncertainty (str): can be `'dropout'` or `'laplace'`. This determines
            how the model will compute uncertainties and covariances.
        
        **kwargs: You can pass in arguments to the destined subclass. So, for
            example, if `model` is a DeepChem model, then `**kwargs` may carry
            any of the arguments accepted by `alien.models.DeepChemRegressor`.
    """
                
    @abstractmethod
    def predict(self, X, return_std_dev=False):
        """
        Applies the model to input(s) X (with the last self.ndim
        axes corresponding to each sample), and returns prediction(s).

        :param return_std_dev: if True, returns a tuple `(prediction, std_dev)`
        """

    def __new__(cls, model=None, X=None, y=None, **kwargs):
        if cls == Regressor:
            if test_if_pytorch(model):
                from .pytorch import PytorchRegressor

                return PytorchRegressor.__new__(
                    PytorchRegressor, model=model, X=X, y=y, **kwargs
                )
            elif test_if_keras(model):
                from .keras import KerasRegressor

                return KerasRegressor.__new__(
                    KerasRegressor, model=model, X=X, y=y, **kwargs
                )
            elif test_if_deepchem(model):
                from .deepchem import DeepChemRegressor

                return DeepChemRegressor.__new__(
                    DeepChemRegressor, model=model, X=X, y=y, **kwargs
                )
            else:
                raise TypeError(
                    f"Regressor doesn't support models of type {model.__class__.__qualname__}"
                )
        else:
            return super().__new__(cls)


class CovarianceRegressor(Regressor):

    def __init__(self, *args, uncertainty=None, use_covariance_for_ensemble=False, **kwargs):
        self.use_covariance_for_ensemble = use_covariance_for_ensemble
        super().__init__(*args, **kwargs)
        if uncertainty is not None:
            self.covariance = getattr(self, 'covariance_' + uncertainty, self.covariance)
            self.std_dev = getattr(self, 'std_dev_' + uncertainty, self.std_dev)

    def covariance(self, X):
        """
        Returns the covariance of the epistemic uncertainty between all 
        rows of X. This is where memory bugs often appear, because of the
        large matrices involved.
        """
        raise NotImplementedError

    # @flatten_batch
    def predict_ensemble(self, X, multiple=1.0):
        """
        Returns a correlated ensemble of predictions for samples X.

        Ensembles are correlated only over the last batch dimension,
        corresponding to axis (-1 - self.ndim) of X. Earlier dimensions
        have no guarantee of correlation.

        :param multiple: standard deviation will be multiplied by this
        """
        return self.predict_samples(X, n=self.ensemble_size, multiple=multiple)

    @get_defaults_from_self
    def predict_samples(self, X, n=1, multiple=1.0, use_covariance_for_ensemble=None):
        if not use_covariance_for_ensemble:
            raise RuntimeError("Using covariance computation to produce ensembles, which is unusual, so we're warning you here. Set `use_covariance_for_ensemble=True` to skip this error.")
        mean, cov = self.predict(X), self.covariance(X)
        return ensemble_from_covariance(mean, multiple * cov, n, self.rng)

    # May want to override this:
    @flatten_batch
    def std_dev(self, X, **kwargs):
        """Returns the (epistemic) standard deviation of the model
        on input `X`."""
        return np.sqrt(self.covariance(X, **kwargs).diagonal())


class EnsembleRegressor(CovarianceRegressor):
    """
    Inherit from EnsembleRegressor if you wish to compute ensembles directly.
    This class provides covariance and prediction for free, given these
    ensembles of predictions.

    Subclasses must implement one of :meth:`predict_ensemble` or
    :meth:`predict_samples`.
    """

    def __init__(self, *args, ensemble_size=40, uncertainty='ensemble', **kwargs):
        super().__init__(*args, ensemble_size=ensemble_size, uncertainty=uncertainty, **kwargs)

    @flatten_batch
    def predict(self, X, return_std_dev=False):
        preds_e = self.predict_ensemble(X)
        preds = preds_e.mean(1)
        if return_std_dev:
            return preds, np.std(preds_e, axis=-1)
        else:
            return preds

    @abstract_group('ensemble')
    def predict_ensemble(self, X, **kwargs):
        """
        Returns an ensemble of predictions.

        :param multiple: standard deviation should be this much larger
        """
        return self.predict_samples(X, n=self.ensemble_size, **kwargs)

    @flatten_batch
    @abstract_group('ensemble')
    def predict_samples(self, X, n=1, **kwargs): #multiple=1.0):
        # Here, we assume `predict_ensemble` has been implemented.
        preds = []
        if n < self.ensemble_size and kwargs.get('multiple', 1) == 1 and hasattr(self, 'models'):
            indices = self.rng.choice(self.ensemble_size, n, replace=False, shuffle=False)
            preds = [self.models[i].predict(X, **kwargs) for i in indices]
        else:
            for j, k in ranges(0, n, self.ensemble_size):
                indices = self.rng.choice(self.ensemble_size, k-j, replace=False, shuffle=False)
                preds.append(self.predict_ensemble(X, **kwargs)[:,indices])
        return join(preds)

    @flatten_batch(degree=2)
    def covariance_ensemble(self, X: ArrayLike):
        """Compute covariance from the ensemble of predictions"""
        return covariance_from_ensemble(self.predict_ensemble(X))

    @flatten_batch
    def std_dev_ensemble(self, X):
        """Returns the (epistemic) standard deviation of the model
        on input `X`."""
        return std_dev_from_ensemble(self.predict_ensemble(X))


class WrappedModel(Model):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def fit_model(self, X=None, y=None, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)


# -------- Some mixin classes for embeddings -------- #

class EmbeddableModel(Model):
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

    def __init__(self, *args, embedding='good', **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'embedding'):
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
            self._embedding_method.extend({
                'explicit':[0],
                'last_layer':[1],
                'input':[2],
                'any':[0,1,2],
                'good':[0,1],
            }.get(m, [m]))
        
        self.find_method()

    method_names = {
        0: ['_embedding', 'embed', 'embeddings'],
        1: ['last_layer_embedding', 'last_layer_embed', 'embed_last_layer', 'last_layer'],
        2: ['input_embedding'],
    }

    def find_method(self):
        for m in self._embedding_method:
            for a in self.method_names[m]:
                if hasattr(self, a):
                    self.embedding = getattr(self, a)
                    return
        
        raise NotImplementedError(f"Could not find an embedding for model of type {type(self)}")

    def input_embedding(self, X):
        try:
            X.shape
        except AttributeError:
            raise TypeError(f"`input_embedding` needs the input to be array-like, but you passed a {type(X)}")
        return X


class LastLayerEmbeddableModel(EmbeddableModel):

    def __init__(self, *args, embedding='last_layer', **kwargs):
        super().__init__(*args, embedding=embedding, **kwargs)

    @abstractmethod
    def last_layer_embedding(self, X):
        """Returns the activations of the last layer before the output."""               


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
            from tensorflow.keras import Model
            assert isinstance(model, Model)
            return True
        except (ImportError, AssertionError):
            pass
    return False
