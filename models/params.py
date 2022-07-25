# Sort'in Hat
# Model Training Parameters
#

import mlflow

from typing import Any, Dict
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

params = {
    "imputer": "knn",
    "scaler": "std",
    "models": [
        {
            "flavor": "sklearn",
            "kind": "linear",
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
        {
            "flavor": "xgboost",
            "kind": "boosted_tree",
            "hyperparams": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,
                "subsample": 1,
                "tree_method": "hist",
                "reg_lambda": 1,  # l2 regularization
                "reg_alpha": 0,  # l2 regularization
            },
        },
        {
            "flavor": "sklearn",
            "kind": "neural_net",
            "hyperparams": {
                "hidden_layer_sizes": [100],
                "solver": "adam",
                "alpha": 0.0001,  # l2 regularization
                "batch_size": "auto",
                "learning_rate_init": 0.3,
                "max_iter": 200,  # no. of epochs
            },
        },
        {
            "flavor": "sklearn",
            "kind": "svm",
            "hyperparams": {
                "kernel": "rbf",
                "gamma": "scale",
                "C": 1.0,
            }
        },
    ],
}

# Feature Extraction
imputers = {"median": SimpleImputer(strategy="median"), "knn": KNNImputer()}

scalers = {"std": StandardScaler(), "minmax": MinMaxScaler()}

# Model Selection
models = {
    "linear": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "boosted_tree": XGBRegressor,
    "neural_net": MLPRegressor,
    "svm": SVR,
}

# MLflow model flavor loggers
model_loggers = {
    "sklearn": mlflow.sklearn.log_model,
    "xgboost": mlflow.xgboost.log_model,
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
