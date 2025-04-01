import os
from typing import List

import numpy as np
import pytest
from pytest import importorskip

torch = importorskip("torch")
tf = importorskip("tensorflow")
dc = importorskip("deepchem")
lgb = importorskip("lightgbm")

nn = torch.nn
keras = tf.keras

SEED = 0
np.random.seed(SEED)

N_SAMPLES = 64
N_FEATURES = 16
N = 23
ENSEMBLE_SIZE = 10
VIRTUAL_ENSEMBLES = 2
N_TREES = 100
N_CLASSES = 5
BATCH_SIZE = 10

DEEPCHEM_DATA_PATH = "tests/sample_data/sample_deepchem_data.csv"
X = np.random.normal(size=(N_SAMPLES, N_FEATURES))
y = np.random.normal(size=N_SAMPLES)
y_class = np.random.randint(N_CLASSES, size=N_SAMPLES)
ensemble_preds = np.random.normal(size=(N_SAMPLES, ENSEMBLE_SIZE))

data = np.vstack([X.T, y]).T

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

    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
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


# Define fixtures
@pytest.fixture(scope="session")
def get_X():
    return X


@pytest.fixture(scope="session")
def get_y():
    return y


@pytest.fixture(scope="session")
def get_y_class():
    return y_class


@pytest.fixture(scope="session")
def get_data():
    return data


@pytest.fixture(scope="session")
def get_ensemble_preds():
    return ensemble_preds


@pytest.fixture(scope="session")
def get_keras_model():
    return keras_model


@pytest.fixture(scope="session")
def get_nn_model():
    return NeuralNetwork()


@pytest.fixture(scope="session")
def get_nn_model_small():
    return NeuralNetwork(input_size=2)


@pytest.fixture(scope="session")
def get_dc_torch_model():
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
    return dc_torch_model


@pytest.fixture(scope="session")
def get_dc_keras_model():
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
    return dc_keras_model
