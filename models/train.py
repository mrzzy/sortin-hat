#
# Sort'in Hat
# Train Model
#

import re
import pandas as pd

from logging import log
from pathlib import Path
from argparse import ArgumentParser

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import KNNImputer

from prepare import load_dataset, segment_dataset


if __name__ == "__main__":
    # parse command line arguments
    parser = ArgumentParser(description="Train Models")

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory to source model training data from.",
    )
    parser.add_argument(
        "models_dir",
        type=Path,
        help="Path to output trained models & metrics as an MLflow filestore",
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

    # load dataset
    df = load_dataset(args.data_dir, args.extract_regex, args.p6_regex)
    print(df.columns.values)
    for level, subject, features_df, targets in segment_dataset(df):
        # skip subjects with too few rows (<20)
        if len(targets) < 20:
            log.warning(
                f"Skipping subject with <20 rows: {subject} - {len(targets)}"
            )
            continue

        # hold out a test set for to faciliate unbiased model evaluation later
        (
            train_features,
            test_features,
            train_targets,
            test_targets,
        ) = train_test_split(features_df, targets, random_state=42, test_size=0.3)

        # build linear model pipeline
        model = Pipeline(
            steps=[
                ("imputer", KNNImputer()),
                ("scale", StandardScaler()),
                ("LR", Ridge(alpha=3e3)),
            ]
        )

        # evaluate model performance with k fold cross validation
        scores_df = pd.DataFrame(
            cross_validate(
                model,
                train_features,
                train_targets,
                scoring=["neg_root_mean_squared_error", "r2"],
                return_train_score=True,
                cv=5,
                n_jobs=-1,
            )
        )
        print("=================== {", subject, "} ====================")
        metrics = [
            metric
            for metric in scores_df.columns
            if "train" in metric or "test" in metric
        ]
        print(scores_df[metrics].mean())
