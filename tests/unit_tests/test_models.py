"""Test a selection of models that best capture the instantiated classes in the models module.
Test output shapes, instance methods (e.g. covariance, linearization, prediction)
Currently tested models are:
    - BayesianRidgeRegressor
    - RandomForestRegressor
    - CatBoostRegressor
    - GaussianProcessRegressor
    - MCDropoutRegressor
    - LightGBMRegressor
    - PytorchRegressor
    - KerasRegressor
"""

import os

import numpy as np
import pytest
from pytest import importorskip

from alien.data import DeepChemDataset, TeachableDataset
from alien.models import DeepChemRegressor, KerasRegressor, PytorchRegressor, Regressor
from alien.models.deepchem import DeepChemKerasRegressor, DeepChemPytorchRegressor
from alien.models.utils import get_base_model

# Skip testing models if torch / tensorflow / deepchem is not installed
torch = importorskip("torch")
tf = importorskip("tensorflow")
dc = importorskip("deepchem")

nn = torch.nn
keras = tf.keras

SEED = 0
np.random.seed(SEED)
N_SAMPLES = 64
N_FEATURES = 16
N = 23
DEEPCHEM_DATA_PATH = "tests/unit_tests/data/sample_deepchem_data.csv"
X = np.random.normal(size=(N_SAMPLES, N_FEATURES))
y = np.random.normal(size=N_SAMPLES)
data = np.vstack([X.T, y]).T
numpy_db = TeachableDataset.from_data(data)
ensemble_size = 13

keras_model = tf.keras.Sequential(
    (
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="gelu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="gelu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    )
)


class NeuralNetwork(nn.Module):
    """Sample Pytorch Neural Network class"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class TestPytorchMCDropoutRegressor:
    model = NeuralNetwork()
    X_tensor = torch.from_numpy(X).type(torch.float32)
    y_tensor = torch.from_numpy(y).type(torch.float32)
    regressor = PytorchRegressor(
        model=model,
        X=X_tensor,
        y=y_tensor,
        ensemble_size=ensemble_size,
        frozen_layers=model.linear_relu_stack[-1],
        uncertainty="dropout",
    )

    def test_fit(self):
        regressor = TestPytorchMCDropoutRegressor.regressor
        regressor.fit(
            X=TestPytorchMCDropoutRegressor.X_tensor,
            y=TestPytorchMCDropoutRegressor.y_tensor,
            batch_size=16,
        )

    def test_covariance(self):
        regressor = TestPytorchMCDropoutRegressor.regressor
        cov_mat = regressor.covariance(TestPytorchMCDropoutRegressor.X_tensor)
        expected_shape = (X.shape[0], X.shape[0])
        assert cov_mat.shape == expected_shape

    def test_ensemble(self):
        regressor = TestPytorchMCDropoutRegressor.regressor
        ensemble_preds = regressor.predict_ensemble(TestPytorchMCDropoutRegressor.X_tensor)
        expected_shape = (
            TestPytorchMCDropoutRegressor.X_tensor.shape[0],
            ensemble_size,
        )
        assert ensemble_preds.shape == expected_shape

    def test_creation(self):
        regressor = Regressor(
            model=self.model, X=X, y=y, ensemble_size=ensemble_size, uncertainty="dropout"
        )
        assert isinstance(regressor, PytorchRegressor)


class TestKerasMCDropoutRegressor:
    model = tf.keras.Sequential(
        (
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation="gelu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation="gelu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        )
    )
    regressor = KerasRegressor(
        model=model, X=X, y=y, ensemble_size=ensemble_size, uncertainty="dropout"
    )

    def test_fit(self):
        regressor = TestKerasMCDropoutRegressor.regressor
        regressor.fit(epochs=5)

    def test_ensemble(self):
        regressor = TestKerasMCDropoutRegressor.regressor
        ensemble_preds = regressor.predict_ensemble(X)
        expected_shape = (X.shape[0], ensemble_size)
        assert ensemble_preds.shape == expected_shape


class TestDeepChemMCDropoutRegressorTorch:
    hyper_params_torch = {
        "n_layers": 2,
        "nb_epoch": 2,
        "nb_epochs_per_cycle": 1,
        "batch_size": 64,
        "predictor_hidden_feats": 50,
        "graph_conv_layers": [20, 20],
        "activation": torch.nn.ReLU(),
        "dropout": 0.1,
        "predictor_dropout": 0.1,
        "patience": 20,
        "learning_rate": 0.001,
    }
    os.makedirs("deepchem_pytorch_models", exist_ok=True)
    dc_torch_model = dc.models.GCNModel(
        1,
        batchnorm=True,
        mode="regression",
        uncertainty=False,
        log_frequency=20,
        model_dir="deepchem_pytorch_models/",
        device="cpu",
        **hyper_params_torch,
    )
    dummy_layer = torch.nn.Identity()
    data = DeepChemDataset.from_csv(
        DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="molgraphconv"
    )

    # pylint: disable=abstract-class-instantiated
    regressor = DeepChemRegressor(
        dc_torch_model,
        data,
        ensemble_size=ensemble_size,
        uncertainty="dropout",
    )

    def test_fit(self):
        regressor = TestDeepChemMCDropoutRegressorTorch.regressor
        regressor.fit()

    def test_predict(self):
        regressor = TestDeepChemMCDropoutRegressorTorch.regressor
        X_pred = TestDeepChemMCDropoutRegressorTorch.data[:3, "X"]
        preds = regressor.predict(X_pred)
        assert preds.shape == (X_pred.shape[0],)

    def test_ensemble(self):
        regressor = TestDeepChemMCDropoutRegressorTorch.regressor
        X_pred = TestDeepChemMCDropoutRegressorTorch.data[:3, "X"]
        ensemble_preds = regressor.predict_ensemble(X_pred)
        expected_shape = (X_pred.shape[0], ensemble_size)
        assert (
            ensemble_preds.shape == expected_shape
        ), f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_covariance(self):
        regressor = TestDeepChemMCDropoutRegressorTorch.regressor
        X_pred = TestDeepChemMCDropoutRegressorTorch.data[:3, "X"]
        cov = regressor.covariance(X_pred)
        assert cov.shape == (X_pred.shape[0], X_pred.shape[0])

    def test_get_train_dataset(self):
        regressor = TestDeepChemMCDropoutRegressorTorch.regressor
        train_np = regressor.get_train_dataset(X=X, y=y)
        assert isinstance(train_np, type(TestDeepChemMCDropoutRegressorTorch.data._to_DC()))
        train_alien = regressor.get_train_dataset(X=TestDeepChemMCDropoutRegressorTorch.data)
        assert isinstance(train_alien, type(TestDeepChemMCDropoutRegressorTorch.data._to_DC()))


class TestDeepChemMCDropoutRegressorKeras:
    hyper_params_keras = {
        "dense_layer_size": 50,
        "graph_conv_layers": [20, 20],
        "dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 0.001,
    }

    os.makedirs("deepchem_models", exist_ok=True)
    dc_keras_model = dc.models.GraphConvModel(
        1,
        batch_normalize=True,
        mode="regression",
        uncertainty=False,
        log_frequency=20,
        model_dir="deepchem_models/",
        **hyper_params_keras,
    )
    nodropout = dc_keras_model.model.graph_convs[-1]

    data = DeepChemDataset.from_csv(
        DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="convmol"
    )

    # pylint: disable=abstract-class-instantiated
    regressor = DeepChemRegressor(
        dc_keras_model,
        data,
        ensemble_size=ensemble_size,
        frozen_layers=nodropout,
        uncertainty="dropout",
    )

    def test_fit(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        regressor.fit()

    def test_predict(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        X_pred = TestDeepChemMCDropoutRegressorKeras.data[:3, "X"]
        preds = regressor.predict(X_pred)
        assert preds.shape == (X_pred.shape[0],)

    def test_ensemble(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        X_pred = TestDeepChemMCDropoutRegressorKeras.data[:3, "X"]
        ensemble_preds = regressor.predict_ensemble(X_pred)
        expected_shape = (X_pred.shape[0], ensemble_size)
        assert (
            ensemble_preds.shape == expected_shape
        ), f"Expected shape {expected_shape}. Got {ensemble_preds.shape}"

    def test_covariance(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        X_pred = TestDeepChemMCDropoutRegressorKeras.data[:3, "X"]
        cov = regressor.covariance(X_pred)
        assert cov.shape == (X_pred.shape[0], X_pred.shape[0])

    def test_get_train_dataset(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        train_np = regressor.get_train_dataset(X=X, y=y)
        assert isinstance(train_np, type(TestDeepChemMCDropoutRegressorKeras.data._to_DC()))
        train_alien = regressor.get_train_dataset(X=TestDeepChemMCDropoutRegressorKeras.data)
        assert isinstance(train_alien, type(TestDeepChemMCDropoutRegressorKeras.data._to_DC()))


class TestPytorchLastLayerLaplaceRegressor:
    model = NeuralNetwork()
    X_tensor = torch.from_numpy(X).type(torch.float32)
    y_tensor = torch.from_numpy(y).type(torch.float32)
    regressor = PytorchRegressor(model=model, X=X_tensor, y=y_tensor, uncertainty="laplace")

    def test_find_last_layer(self):
        regressor = TestPytorchLastLayerLaplaceRegressor.regressor
        _ = regressor.find_last_layer(TestPytorchLastLayerLaplaceRegressor.X_tensor)
        assert isinstance(
            regressor.last_layer, nn.Linear
        ), f"Last layer should be nn.Linear, not {type(regressor.last_layer)}"

    def test_embeddings(self):
        regressor = TestPytorchLastLayerLaplaceRegressor.regressor
        _, embd_1 = regressor.predict_with_embedding(TestPytorchLastLayerLaplaceRegressor.X_tensor)
        embd_2 = regressor.embedding(TestPytorchLastLayerLaplaceRegressor.X_tensor)
        assert (
            (embd_1 == embd_2).all().item()
        ), "embedding(*) and predict_with_embedding(*) should return the same embeddings."

    def test_predict(self):
        regressor = TestPytorchLastLayerLaplaceRegressor.regressor
        preds_1, _ = regressor.predict_with_embedding(
            TestPytorchLastLayerLaplaceRegressor.X_tensor
        )
        preds_2 = regressor.predict(TestPytorchLastLayerLaplaceRegressor.X_tensor)
        assert (
            (preds_1 == preds_2).all().item()
        ), "predict (*) and predict_with_embedding(*) should return the same output."


class TestDeepChemKerasLaplaceRegressor:
    hyper_params_keras = {
        "dense_layer_size": 50,
        "graph_conv_layers": [20, 20],
        "dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 0.001,
    }

    os.makedirs("deepchem_models", exist_ok=True)
    dc_keras_model = dc.models.GraphConvModel(
        1,
        batch_normalize=True,
        mode="regression",
        uncertainty=False,
        log_frequency=20,
        model_dir="deepchem_models/",
        **hyper_params_keras,
    )
    nodropout = dc_keras_model.model.graph_convs[-1]

    data = DeepChemDataset.from_csv(
        DEEPCHEM_DATA_PATH, X="SMILES", y="y_exp", featurizer="convmol"
    )

    # pylint: disable=abstract-class-instantiated
    regressor = DeepChemRegressor(dc_keras_model, data, uncertainty="laplace")

    def test_fit(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        regressor.fit()

    def test_predict(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        X_pred = TestDeepChemMCDropoutRegressorKeras.data[:3, "X"]
        preds = regressor.predict(X_pred)
        assert preds.shape == (X_pred.shape[0],)

    def test_covariance(self):
        regressor = TestDeepChemMCDropoutRegressorKeras.regressor
        X_pred = TestDeepChemMCDropoutRegressorKeras.data[:3, "X"]
        cov = regressor.covariance(X_pred)
        assert cov.shape == (X_pred.shape[0], X_pred.shape[0])


class TestLaplaceRegressor:
    def test_creation(self):
        regressor = Regressor(
            model=NeuralNetwork(),
            X=X,
            y=y,
            uncertainty="laplace",
        )
        assert isinstance(regressor, PytorchRegressor)


def test_get_base_model():
    pt_model = NeuralNetwork()
    al_model = Regressor(pt_model)
    assert get_base_model(al_model) == pt_model
    assert get_base_model(al_model, framework="torch") == pt_model
    with pytest.raises(ValueError):
        assert get_base_model(al_model, framework="keras")
