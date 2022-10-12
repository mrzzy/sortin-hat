#
# sortin-hat
# Pipeline
# Airflow DAG
#

from typing import Dict, Iterable, Tuple, cast

import pandas as pd
from airflow.decorators import dag, task
from airflow.models.connection import Connection
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone
from ray import tune
from sklearn.metrics import mean_squared_error, r2_score

from extract import extract_features, featurize_dataset, vectorize_features
from model import MODELS
from transform import unpivot_subjects

TIMEZONE = "Asia/Singapore"
DAG_ID = "sortin-hat-pipeline"


def pd_storage_opts(gcp_connection_id: str) -> Dict:
    """Build Pandas storage options GCS I/O with the GCP connection specified by id."""
    # extract gcp service account json key path from airflow gcp connection
    return {
        "token": (
            Connection.get_connection_from_secrets(gcp_connection_id).extra_dejson[
                "extra__google_cloud_platform__key_path"
            ]
        )
    }


def local_year(timestamp: DateTime, local_tz: str = TIMEZONE) -> int:
    """Obtain the year of the given datetime in the local time zone."""
    return timestamp.astimezone(timezone(local_tz)).year


def load_dataset(
    gcp_connection_id: str,
    datasets_bucket: str,
    dataset_prefix: str,
    years: Iterable[int],
) -> pd.DataFrame:
    """
    Load the yearly-partitioned Sortin-Hat Dataset as single DataFrame.
    'years' specifies which year's paritions should be included in the DataFrame.
    """

    def add_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        df["Year"] = year
        return df

    return pd.concat(
        [
            add_year(
                pd.read_parquet(
                    f"gs://{datasets_bucket}/{dataset_prefix}/{year}.pq",
                    storage_options=pd_storage_opts(gcp_connection_id),
                ),
                year,
            )
            for year in years
        ]
    )


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the given dataset into a input features & target series."""
    # Prediction Target
    TARGET = "Score"
    return (df[[column for column in df.columns if column != TARGET]], df[TARGET])


@dag(
    dag_id=DAG_ID,
    description="Sortin-hat ML Pipeline",
    # each dag run handles a year-sized data interval from start_date
    start_date=datetime(2016, 1, 1, tz=timezone(TIMEZONE)),
    schedule_interval="@yearly",
)
def pipeline(
    raw_bucket: str = "sss-sortin-hat-raw-data",
    raw_s4_prefix: str = "Sec4_Cohort",
    raw_p6_prefix: str = "P6_Screening",
    datasets_bucket: str = "sss-sortin-hat-datasets",
    models_bucket: str = "sss-sortin-hat-models",
    timezone_str: str = TIMEZONE,
    gcp_connection_id="google_cloud_default",
    mlflow_experiment_id: str = "sss-score-prediction",
):
    f"""
    # Sortin-hat ML Pipeline
    End to End Pipeline for training Sortin-hat ML models for predicting student scores.

    ## Prerequisites
    ### Connections
    Expects a GCP connection to be configured with the under the id `gcp_connection_id`
    with extra for keyfile json set.

    ### Infrastructure
    GCS buckets `raw_bucket`, `datasets_bucket` & `models_bucket` should be created
    beforehand.

    ### Data Source
    The pipeline takes in as data source 2 kinds of Excel Spreadsheets stored
    in the `raw_bucket` GCS bucket, partitioned by year:
    - Sec 4 Cohort Sendout, stored as `raw_s4_prefix/<YEAR>.xlsx`
    - Optional P6 Screening Template, stored as `raw_p6_prefix/<YEAR>.xlsx`

    > An assumption is made that all dates are expressed in the Asia/Singapore time zone.

    ## Machine Learning
    ### Cross Validation
    Sound ML practice dictates that we train model on a Training set &
    cross validate our result on an hold out Test set:
    - Training : All data leading up, but excluding the current year's partition.
    - Test set: The current year's partition.

    > Current year is defined as the year the data interval the pipeline DAG run
    is processing.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the
    `models_bucket` GCS bucket.
    """

    @task(
        task_id="transform_dataset",
    )
    def transform_dataset(
        gcp_connection_id: str,
        raw_bucket: str,
        raw_s4_prefix: str,
        raw_p6_prefix: str,
        datasets_bucket: str,
        dataset_prefix: str,
        timezone_str: str,
        data_interval_start: DateTime = cast(DateTime, None),
    ):
        # TODO(mrzzy): unit test
        """
        Transform the Data source Excel Spreadsheets into Parquet Dataset of
        ML model tailored Features.
        Both data source & dataset are partitioned by cohort year
        Writes the Parquet dataset in GCS.
        """
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        from clean import clean_extract, clean_p6
        from extract import extract_features
        from transform import suffix_subject_level

        # load & Clean data from excel spreadsheet(s)
        year = local_year(data_interval_start)
        storage_opts = pd_storage_opts(gcp_connection_id)
        df = clean_extract(
            pd.read_excel(
                f"gs://{raw_bucket}/{raw_s4_prefix}/{year}.xlsx",
                storage_options=storage_opts,
            )
        )

        # apply transforms to data
        df = suffix_subject_level(df, year)
        df = unpivot_subjects(df, year)

        # merge in cleaned p6 data if it exists
        gcs = GCSHook(gcp_connection_id)
        if gcs.exists(raw_bucket, f"{raw_p6_prefix}/{year}.xlsx"):
            p6_df = clean_p6(
                pd.read_excel(
                    # header=1: headers are stored in p6 data on the second row
                    f"gs://{raw_bucket}/{raw_p6_prefix}/{year}.xlsx",
                    storage_options=storage_opts,
                    header=1,
                )
            )
            df = pd.merge(df, p6_df, how="left", on="Serial number")
        # write transformed dataset as compressed parquet file
        df.to_parquet(
            f"gs://{datasets_bucket}/{dataset_prefix}/{year}.pq",
            storage_options=storage_opts,
        )

    @task(depends_on_past=True)
    def train_tuned_model(
        model_name: str,
        gcp_connection_id: str,
        datasets_bucket: str,
        dataset_prefix: str,
        timezone_str: str,
        mlflow_experiment_id: str,
        start_date: DateTime = cast(DateTime, None),
        data_interval_start: DateTime = cast(DateTime, None),
    ):
        # TODO(mrzzy): run on ray cluster
        f"""
        Trains multiple {model_name} models on the Training set with different hyperparameters
        in order to experiment with hyperparameter combinations:
        - Feature Preprocessing methods used.
        - Model-specific Hyperparameters.

        To avoid time leakage, dataset split is selected by cohort year (relative
        to the DAG processed data interval's year):
        - Training Set meant for fitting models includes all data up to
            & including the 3rd latest cohort year.
        - Validation Set consists of the 2nd latest cohort year. It is used
            for hyperparameter tuning.
        - Test Set consists the latest cohort year. It is used for unbiased
            estimate of final model performance.
        """
        # verify we have enough partitions to split dataset into train/validate/test
        begin_year, current_year = local_year(start_date), local_year(
            data_interval_start
        )

        n_partitions = current_year - begin_year
        if n_partitions < 3:
            raise RuntimeError(
                f"DAG Data Interval too small: expected >3 partitions, got {n_partitions}"
            )

        def objective(params: Dict):
            # load train, validate & test datasets
            load_years = lambda years: load_dataset(
                gcp_connection_id, datasets_bucket, dataset_prefix, years
            )
            train_features, train_targets = featurize_dataset(
                load_years(range(begin_year, current_year - 1))
            )
            validate_features, validate_targets = featurize_dataset(
                load_years([current_year - 1])
            )
            test_features, test_targets = featurize_dataset(load_years([current_year]))

            # train model on training set
            model = MODELS[model_name].build(params)
            model.fit(train_features, train_targets)

            # evaluate model fit using metrics
            metrics = {
                "rmse": lambda features, targets: mean_squared_error(
                    features, targets, squared=False
                ),
                "r2": r2_score,
            }
            subsets = {
                "train": (train_features, train_targets),
                "validate": (validate_features, validate_targets),
                "test": (test_features, test_targets),
            }
            results = {
                f"{subset}_{metric}": metric_fn(*data)
                for subset, data in subsets.items()
                for metric, metric_fn in metrics.items()
            }
            tune.report(**results)

    dataset_prefix = "dataset"
    transform_dataset(
        gcp_connection_id,
        raw_bucket,
        raw_s4_prefix,
        raw_p6_prefix,
        datasets_bucket,
        dataset_prefix,
        timezone_str,
    )
