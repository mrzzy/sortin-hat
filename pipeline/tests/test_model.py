#
# sortin-hat
# Pipeline
# ML Models: Unit Tests
#

from shutil import rmtree
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Dict

import pytest
from ray.tune import Tuner
from sklearn.linear_model import ElasticNet

from model import LinearRegression


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
