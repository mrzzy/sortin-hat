#
# sortin-hat
# Pipeline
# Airflow DAG
#

import os
from os.path import dirname
from typing import cast

import pandas as pd
from airflow.decorators import dag, task
from airflow.models.connection import Connection
from airflow.models.dag import DAG
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone
from ray.air.config import RunConfig

from clean import P6_COLUMNS

TIMEZONE = "Asia/Singapore"
DAG_ID = "sortin-hat-pipeline"
DAG_START_DATE = datetime(2016, 1, 1, tz=timezone(TIMEZONE))


def gcp_key_path(gcp_connection_id: str) -> str:
    """Extract the GCP service account json key from the airflow GCP connection specified by id."""
    return Connection.get_connection_from_secrets(gcp_connection_id).extra_dejson[
        "extra__google_cloud_platform__key_path"
    ]


def local_year(timestamp: DateTime, local_tz: str = TIMEZONE) -> int:
    """Obtain the year of the given datetime in the local time zone."""
    return timestamp.astimezone(timezone(local_tz)).year


@dag(
    dag_id=DAG_ID,
    description="Sortin-hat ML Pipeline",
    # each dag run handles a year-sized data interval from start_date
    start_date=DAG_START_DATE,
    schedule_interval="@yearly",
)
def pipeline(
    raw_bucket: str = "sss-sortin-hat-raw-data",
    raw_s4_prefix: str = "Sec4_Cohort",
    raw_p6_prefix: str = "P6_Screening",
    datasets_bucket: str = "sss-sortin-hat-datasets",
    tune_n_trails: int = 1,
    ray_address: str = "ray://ray:10001",
    mlflow_tracking_url: str = "http://mlflow:5000",
    mlflow_experiment: str = DAG_ID,
    gcp_connection_id="google_cloud_default",
):
    f"""
    # Sortin-hat ML Pipeline
    End to End Pipeline for training Sortin-hat ML models for predicting student scores.

    ## Prerequisites
    ### Connections
    Expects a GCP connection to be configured with the under the id `gcp_connection_id`
    with extra for keyfile json set.

    ### Infrastructure
    The Pipeline expects the following infrastructure to be deployed beforehand.
    - GCS buckets `raw_bucket`, `datasets_bucket`
    - Ray Cluster listening at `ray_address`.
    - MLFlow Tracking server listening at `mlflow_tracking_url`.

    ### Data Source
    The pipeline takes in as data source 2 kinds of Excel Spreadsheets stored
    in the `raw_bucket` GCS bucket, partitioned by year:
    - Sec 4 Cohort Sendout, stored as `raw_s4_prefix/<YEAR>.xlsx`
    - Optional P6 Screening Template, stored as `raw_p6_prefix/<YEAR>.xlsx`

    > An assumption is made that all dates are expressed in the Asia/Singapore time zone.

    ## Machine Learning
    Performs `tune_n_trails` no. of ML Model training/evaluation runs to tune
    hyperparameters logging. Increasing `tune_n_trails` will conduct a more extensive
    hyperparameter search to find better hyperparameters. However, it comes at
    the trade off of needing more computing power to run these trials.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the
    MLFlow tracking server.
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
        data_interval_start: DateTime = cast(DateTime, None),
    ):
        """
        Transform the Data source Excel Spreadsheets into Parquet Dataset of
        ML model tailored Features.
        Both data source & dataset are partitioned by cohort year
        Writes the Parquet dataset in GCS.
        """
        # imports done within tasks done to speed up dag definition import times
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        from clean import clean_extract, clean_p6
        from transform import suffix_subject_level, unpivot_subjects

        # configure GCP credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key_path(gcp_connection_id)

        # load & Clean data from excel spreadsheet(s)
        year = local_year(data_interval_start)
        df = clean_extract(
            pd.read_excel(
                f"gs://{raw_bucket}/{raw_s4_prefix}/{year}.xlsx",
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
                    header=1,
                )
            )
            df = pd.merge(df, p6_df, how="left", on="Serial number")
        else:
            # insert empty columns to stand in for missing P6 screening data
            df[P6_COLUMNS] = pd.NA
        # write transformed dataset as compressed parquet file
        df.to_parquet(
            f"gs://{datasets_bucket}/{dataset_prefix}/{year}.pq",
        )

    @task(
        task_id="train_tuned_model",
        # task needs at historical yearly dataset partitions as model training data.
        depends_on_past=True,
    )
    def train_tuned_model(
        model_name: str,
        n_trials: int,
        datasets_bucket: str,
        dataset_prefix: str,
        ray_address: str,
        mlflow_tracking_url: str,
        mlflow_experiment: str,
        dag: DAG = cast(DAG, None),
        data_interval_start: DateTime = cast(DateTime, None),
    ):
        f"""
        Trains {n_trials} trials of the {model_name} model on the Training set with
        different hyperparameters in order to experiment with hyperparameter combinations.

        To avoid time leakage, dataset split is selected by cohort year (relative
        to the DAG processed data interval's year):
        - Training Set meant for fitting models includes all data up to
            & including the 3rd latest cohort year.
        - Validation Set consists of the 2nd latest cohort year. It is used
            for hyperparameter tuning.
        - Test Set consists the latest cohort year. It is used for unbiased
            estimate of final model performance.

        Evaluation of the model is performed with the following metrics on all
        3 dataset splits:
        - Mean Squared Error (MSE) is used to perform hyperparameter tuning.
        - Root MSE provides human interpretable alternative to MSE. It can be
            interpreted as the average difference between predicted & actual scores.
        - R2 measures the model's quality of fit by calculating the % of variance
            in the data accounted for by the model.
        """
        # imports done within tasks done to speed up dag definition import times
        import ray
        from ray import tune
        from ray.tune.integration.mlflow import MLflowLoggerCallback
        from ray.tune.tune_config import TuneConfig

        from model import MODELS
        from train import run_objective

        # verify we have enough partitions to split dataset into train/validate/test
        begin_year = local_year(cast(DateTime, dag.start_date))
        current_year = local_year(data_interval_start)

        n_partitions = current_year - begin_year + 1
        if n_partitions < 3:
            print(
                f"Skipping: DAG Data Interval too small: expected >=3 partitions, got {n_partitions}"
            )
            return

        # log tuning experiment to mlflow with callback
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=mlflow_tracking_url,
            experiment_name=mlflow_experiment,
            tags={"model": model_name},
            save_artifact=True,
        )
        # find optimal model hyperparameters with ray tune via experiments
        ray.init(
            ray_address,
            runtime_env={
                "working_dir": dirname(__file__),
            },
        )
        params = MODELS[model_name].param_space()
        params.update(
            {
                "datasets_bucket": datasets_bucket,
                "dataset_prefix": dataset_prefix,
                "begin_year": begin_year,
                "current_year": current_year,
                "model_name": model_name,
            }
        )

        tuner = tune.Tuner(
            trainable=run_objective,
            param_space=params,
            tune_config=TuneConfig(
                metric="validate_mse",
                num_samples=n_trials,
            ),
            run_config=RunConfig(callbacks=[mlflow_callback]),
        )
        tuner.fit()

    # define task dependencies in dag
    dataset_prefix = "dataset"
    dataset_op = transform_dataset(
        gcp_connection_id,
        raw_bucket,
        raw_s4_prefix,
        raw_p6_prefix,
        datasets_bucket,
        dataset_prefix,
    )
    train_op = train_tuned_model(
        model_name="Linear Regression",
        n_trials=tune_n_trails,
        datasets_bucket=datasets_bucket,
        dataset_prefix=dataset_prefix,
        ray_address=ray_address,
        mlflow_tracking_url=mlflow_tracking_url,
        mlflow_experiment=mlflow_experiment,
    )

    dataset_op >> train_op  # type: ignore


pipeline_dag = pipeline()
