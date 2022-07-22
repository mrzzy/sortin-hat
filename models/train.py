#
# Sort'in Hat
# Train Model
#

import re
import pandas as pd
import mlflow as ml
import logging as log

from pathlib import Path
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline

from prepare import load_dataset, segment_dataset
from params import params, hydrate, model_loggers


if __name__ == "__main__":
    log.basicConfig(
        level=log.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    )
    l = log.getLogger(__file__)

    # parse command line arguments
    parser = ArgumentParser(description="Train Models")

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory to source model training data from.",
    )
    parser.add_argument(
        "mlflow_url",
        type=str,
        help="MLFlow URL to output train ML model, hyperparameters & metrics to.",
        default=ml.get_tracking_uri(),
    )
    parser.add_argument(
        "--extract-regex",
        type=re.compile,
        default=r"Template for Data Extraction \(DL Project\) (?P<year>\d{4}) Sec4 cohort\s+Sendout.xlsx",
        help="Regular expression matching 'Data Extraction' Excel files.",
    )
    parser.add_argument(
        "--p6-regex",
        type=re.compile,
        default=r"P6 Screening Master Template \(\d\d? \w\w\w \d{4}\)_Last Sendout.xlsx",
        help="Regular expression matching 'P6 Screening' Excel files.",
    )

    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise ValueError("Expected data_dir to be a path to valid directory.")

    # configure mlflow to output to mlflow tracking server
    ml.set_registry_uri(args.mlflow_url)
    ml.set_tracking_uri(args.mlflow_url)

    # load dataset
    df = load_dataset(args.data_dir, args.extract_regex, args.p6_regex)

    for level, subject, features_df, targets in segment_dataset(df):
        # skip subjects with too few rows (<20)
        if len(targets) < 20:
            l.warning(
                f"Skipping subject with <20 rows: subject={subject} rows={len(targets)}"
            )
            continue

        experiment = ml.set_experiment(subject)

        # train all model / subject combinations
        components = hydrate(params)
        for model, model_params in zip(components["models"], params["models"]):
            with ml.start_run(experiment_id=experiment.experiment_id) as run:
                # log training parameters to mlflow
                for param in ["imputer", "scaler"]:
                    ml.log_param(param, params[param])
                ml.log_params(model_params)

                # tag run with dataset segment & log model training / evaluation run parameters
                ml.set_tags(
                    {"level": f"S{level}", "subject": subject, "model": model["kind"]}
                )
                # hold out a test set for to faciliate unbiased model evaluation later
                (
                    train_features,
                    test_features,
                    train_targets,
                    test_targets,
                ) = train_test_split(
                    features_df,
                    targets,
                    random_state=42,
                    test_size=0.3,
                )

                # build model pipeline
                pipeline = Pipeline(
                    steps=[(k, components[k]) for k in ["imputer", "scaler"]]
                    + [
                        ("model", model["model"]),
                    ],
                )
                pipeline.fit(train_features, train_targets)

                # save trained model to mlflow
                model_loggers["sklearn"](pipeline["imputer"], "imputer")
                model_loggers["sklearn"](pipeline["scaler"], "scaler")
                model["logger"](pipeline["model"], model["flavor"])

                # evaluate model performance with k fold cross validation
                metrics_df = pd.DataFrame(
                    cross_validate(
                        pipeline,
                        train_features,
                        train_targets,
                        scoring=["neg_root_mean_squared_error", "r2"],
                        return_train_score=True,
                        cv=5,
                        n_jobs=-1,
                    )
                ).mean()
                ml.log_metrics(metrics_df.to_dict())
                l.info(f"Trained Model with metrics: \n{metrics_df}")
