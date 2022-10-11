#
# sortin-hat
# Pipeline
# ML Models
#

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
from mlflow import sklearn
from numpy.typing import NDArray
from ray import tune
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet


class Model(ABC):
    """Abstract ML Model interface implemented by all models."""

    @staticmethod
    @abstractmethod
    def param_space() -> Dict:
        """Defining hyperparameters a search space for the model.

        Returns:
            Dictionary representing he hyperparameter search space in terms of
            Ray Tune's Search Space API.
        """
        pass

    @classmethod
    @abstractmethod
    def build(cls, params: Dict) -> None:
        """Build the model with the given dictionary of hyperparameters."""
        pass

    @abstractmethod
    def fit(self, features: NDArray[np.float_], labels: NDArray[np.float_]):
        """Fit the model to the given training batch features & labels."""
        pass

    @abstractmethod
    def predict(self, features: NDArray[np.float_]) -> NDArray[np.float_]:
        """Make a batch of predictions using the given input features."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Store the Model at the given path in the MLFlow model's format."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """Restore a Model from the MLFlow model at the given path."""
        pass


class LinearRegression(Model):
    """Linear Regression model based on Sckit-learn's ElasticNet."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def param_space() -> Dict:
        return  # TODO:

    @classmethod
    def build(self, params: Dict):
        self.model = ElasticNet(
            alpha=params["l2_regularize"],
            l1_ratio=params["l1_regularize"],
        )

    def fit(self, features: NDArray[np.float_], labels: NDArray[np.float_]):
        self.model.fit(features, labels)

    def predict(self, features: NDArray[np.float_]) -> NDArray[np.float_]:
        return self.model.predict(features)

    def save(self, path: Path):
        sklearn.save_model(self.model, path.as_posix())
        self.model.get_params()

    @classmethod
    def load(cls, path: Path):
        return cls(sklearn.load_model(path.as_posix()))
