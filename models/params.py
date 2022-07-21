#
# Sort'in Hat
# Model Training Parameters
#

import mlflow

from typing import Any, Dict
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler


params = {
    "imputer": "knn",
    "scaler": "std",
    "model": {"flavor": "sklearn","kind": "lr_ridge", "hyperparams": {"alpha": 3e3}},
}

# Feature Extraction
imputers = {"median": SimpleImputer(strategy="median"), "knn": KNNImputer()}

scalers = {"std": StandardScaler(), "minmax": MinMaxScaler()}

# Model Selection
models = {"lr_ridge": Ridge}

# MLflow model flavor loggers
model_loggers = {
    "sklearn": mlflow.sklearn.log_model,
}


def hydrate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate the given training parameters with actual objects.

    Args:
        params:
            Parameters to hydrate with actual objects
    Returns:
        Given parameters with elements replaced with actual objects
    """
    components = {
        "imputer": imputers[params["imputer"]],
        "scaler": scalers[params["scaler"]],
    }

    # build model with given hyperpameters
    Model = models[params["model"]["kind"]]
    components["model"] = Model(**params["model"]["hyperparams"])
    components["log_model"] = model_loggers[params["model"]["flavor"]]

    return components
