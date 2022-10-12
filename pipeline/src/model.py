#
# sortin-hat
# Pipeline
# ML Models
#

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from mlflow import sklearn
from numpy.typing import NDArray
from ray import tune
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
    def save(self, dir_path: str):
        """Store the Model in the directory given path in the MLFlow model's format."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str):
        """Restore a Model from the direction at given path in MLFlow Model format."""
        pass


class LinearRegression(Model):
    """Linear Regression model based on Sckit-learn's ElasticNet."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def param_space() -> Dict:
        return {
            "l2_regularization": tune.loguniform(1e-6, 1),
            "l1_regularization": tune.loguniform(1e-6, 1),
        }

    @classmethod
    def build(cls, params: Dict):
        return cls(
            ElasticNet(
                alpha=params["l2_regularization"],
                l1_ratio=params["l1_regularization"],
            )
        )

    def fit(self, features: NDArray[np.float_], labels: NDArray[np.float_]):
        self.model.fit(features, labels)

    def predict(self, features: NDArray[np.float_]) -> NDArray[np.float_]:
        return self.model.predict(features)

    def save(self, dir_path: str):
        sklearn.save_model(self.model, dir_path)
        self.model.get_params()

    @classmethod
    def load(cls, dir_path: str):
        return cls(sklearn.load_model(dir_path))


MODELS = {"Linear Regression": LinearRegression}
