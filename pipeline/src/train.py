#
# sortin-hat
# Pipeline
# Training Objective
#

import os
from typing import Dict, Iterable

import pandas as pd
from ray import tune
from sklearn.metrics import mean_squared_error, r2_score

from extract import featurize_dataset
from model import MODELS, evaluate_model


def load_dataset(
    datasets_bucket: str,
    dataset_prefix: str,
    years: Iterable[int],
) -> pd.DataFrame:
    """
    Load the yearly-partitioned Sortin-Hat Dataset as single DataFrame.
    'years' specifies which year's partitions should be included in the DataFrame.
    """

    def add_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        df["Year"] = year
        return df

    return pd.concat(
        [
            add_year(
                pd.read_parquet(
                    f"gs://{datasets_bucket}/{dataset_prefix}/{year}.pq",
                ),
                year,
            )
            for year in years
        ]
    )


def run_objective(params: Dict):
    """
    Define objective function for hyperparameter optimization to optimize.
    Expects GCP credentials to be configured via GOOGLE_APPLICATION_CREDENTIALS env var.
    """
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise LookupError(
            "Expected GCP credentials to be configured via GOOGLE_APPLICATION_CREDENTIALS"
            " environment variable."
        )

    # load train, validate & test datasets
    load_years = lambda years: load_dataset(
        params["datasets_bucket"], params["dataset_prefix"], years
    )
    train_features, train_targets = featurize_dataset(
        load_years(range(params["begin_year"], params["current_year"] - 1))
    )
    validate_features, validate_targets = featurize_dataset(
        load_years([params["current_year"] - 1])
    )
    test_features, test_targets = featurize_dataset(
        load_years([params["current_year"]])
    )

    # train model on training set
    model = MODELS[params["model_name"]].build(params)
    model.fit(train_features, train_targets)

    # evaluate model fit using metrics
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "rmse": lambda features, targets: mean_squared_error(
            features, targets, squared=False
        ),
    }
    tune.report(
        **evaluate_model(model, metrics, (train_features, train_targets), "train"),
        **evaluate_model(
            model, metrics, (validate_features, validate_targets), "validate"
        ),
        **evaluate_model(model, metrics, (test_features, test_targets), "test"),
    )
