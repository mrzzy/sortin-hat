#
# sortin-hat
# Pipeline
# ML Models: Unit Tests
#

from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from ray.tune import Tuner
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from model import LinearRegression, Model, evaluate_model


@pytest.fixture
def mock_model() -> Model:
    """Mock model that returns a vector of 5 zeros in its predictions."""

    class MockModel(Model):
        def __init__(self):
            pass

        @staticmethod
        def param_space() -> Dict:
            return {}

        @classmethod
        def build(cls, _: Dict):
            pass

        def fit(self, _: NDArray[np.float_], __: NDArray[np.float_]):
            pass

        def predict(self, _: NDArray[np.float_]) -> NDArray[np.float_]:
            return np.zeros((5,))

        def save(self, _: str):
            pass

        @classmethod
        def load(cls, _: str) -> Model:
            return cls()

    return MockModel()


class TestLinearRegression:
    @pytest.mark.integration
    def test_param_space_build(self):
        def train(params: Dict) -> float:
            LinearRegression.build(params)
            # mock loss by returning constant value
            return 0.5

        # check: ray's Tuner able to parse param space & LinearRegression can b
        # built from hyperparameters
        tuner = Tuner(train, param_space=LinearRegression.param_space())
        tuner.fit()

    @pytest.mark.integration
    def test_save_load_model(self):
        model = LinearRegression(ElasticNet())
        # check: model can save & load from a directory
        dir_path = mkdtemp()
        model.save(dir_path)
        LinearRegression.load(dir_path)
        rmtree(dir_path)


@pytest.mark.unit
def test_evaluate_model(mock_model: Model):
    n_rows, prefix = 5, "test"
    results = evaluate_model(
        model=mock_model,
        metrics={"mse": mean_squared_error},
        data=(
            pd.DataFrame(
                {
                    "A": np.ones(n_rows),
                    "B": np.ones(n_rows),
                }
            ),
            np.zeros((n_rows,)),
        ),
        prefix=prefix,
    )

    # check: result key's prefixed
    assert all(["test" == key[: len(prefix)] for key in results.keys()])
    # check: metric evaluation results
    assert results["test_mse"] == 0.0
