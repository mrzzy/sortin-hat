#
# Sort'in Hat
# Model Training Parameters
#

import mlflow

from typing import Any, Dict
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


params = {
    "imputer": "knn",
    "scaler": "std",
    "models": [
        {
            "flavor": "sklearn",
            "kind": "lr_elasticnet",
            "hyperparams": {
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "random_state": 42,
            },
        },
        {
            "flavor": "sklearn",
            "kind": "decision_tree",
            "hyperparams": {
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
        },
    ],
}

# Feature Extraction
imputers = {"median": SimpleImputer(strategy="median"), "knn": KNNImputer()}

scalers = {"std": StandardScaler(), "minmax": MinMaxScaler()}

# Model Selection
models = {
    "lr_elasticnet": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
}

# MLflow model flavor loggers
model_loggers = {
    "sklearn": mlflow.sklearn.log_model,
}


def hydrate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate the given training parameters into training component objects.

    Args:
        params:
            Parameters to hydrate with actual objects.
    Returns:
        Given parameters with elements replaced with actual objects.
    """
    components = {
        "imputer": imputers[params["imputer"]],
        "scaler": scalers[params["scaler"]],
    }

    # build model with given hyperparameters
    def build_model(config):
        Model = models[config["kind"]]
        return {
            "kind": config["kind"],
            "flavor": config["flavor"],
            "model": Model(**config["hyperparams"]),
            "logger": model_loggers[config["flavor"]],
        }

    components["models"] = [build_model(config) for config in params["models"]]

    return components
