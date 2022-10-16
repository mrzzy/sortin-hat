#
# sortin-hat
# Pipeline
# ML Models
#

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Type

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
            Ray Tune's Search Space API. Parameter keys used should be parsable by build().
        """
        pass

    @classmethod
    @abstractmethod
    def build(cls, params: Dict) -> "Model":
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
    def load(cls, dir_path: str) -> "Model":
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
    def build(cls, params: Dict) -> Model:
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
    def load(cls, dir_path: str) -> Model:
        return cls(sklearn.load_model(dir_path))


""" Metric that takes in expected values in first arg & predictions in second. """
Metric = Callable[[NDArray[np.float_], NDArray[np.float_]], float]


def evaluate_model(
    model: Model,
    metrics: Dict[str, Metric],
    data: Tuple[NDArray[np.float_], NDArray[np.float_]],
    prefix: str = "",
):
    """Evaluate the given model with the given metrics on the given subset of data.

    Args:
        model:
            The model to evaluate.
        metrics:
            Dictionary of metric names to metric functions.
        data:
            Dataset to evaluate on, as a tuple of featrue vectors & target values.
        prefix:
            Optional. Prefix to add to keys in the result dictionary.
    Returns:
        Dictionary containing the results of evaluating metrics on model with the given data.
    """
    features, targets = data
    predictions = model.predict(features)

    return {
        f"{prefix}{metric}": metric_fn(targets, predictions)
        for metric, metric_fn in metrics.items()
    }


MODELS = {"Linear Regression": LinearRegression}  # type: Dict[str, Type[Model]]
