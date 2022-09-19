#
# sortin-hat
# Data & ML Pipeline
# Airflow DAG
#

from os import path
from typing import Optional

import pandas as pd
from airflow.decorators import dag, task
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz.timezone import UTC

from prepare import prepare_extract, prepare_p6

config = {
    "buckets": {
        "raw_data": {
            "name": "sss-sortin-hat-raw-data",
            "sec4_prefix": "Sec4_Cohort",
            "p6_prefix": "P6_Screening",
        },
        "dataset": {
            "name": "sss-sortin-hat-datasets",
            "scores_prefix": "scores",
        }
    }
}

@dag(
    dag_id="sortin-hat-pipeline",
    description="Data & ML Pipeline producing Sortin-hat ML models",

    # each dag run handles a year-sized data interval from start_date
    start_date=datetime(2016, 1, 1, tz="Asia/Singapore"),
    schedule_interval="@yearly",

    # task defaults
    default_args = {
        "email_on_failure": True,
        "email": "program.nom@gmail.com",
    }
)
def pipeline():
    """
    # Sortin-hat: Data & ML Pipeline
    End to End Pipeline for preparing data, training & evaluting ML models.

    ## Input
    The Data & ML Pipeline expects the following spreadsheets as inputs:
    - Sec 4 Cohort Sendout: stored under `<SEC4_PREFIX>/<YEAR>.xlsx`
    - P6 Screening Template: stored under `<P6_PREFIX>/<YEAR>.xlsx`
    > `<SEC4_PREFIX>` & `<P6_PREFIX>` can are configured with
    > the `bucket.raw_data.sec4_prefix` & `bucket.raw_data.p6_prefix`.

    The Excel spreadsheet should be partitioned by year & stored in the 
    `bucket.raw_data.name` GCS Bucket.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the 
    `bucket.models.name` GCS bucket.

    """
    @task(
        task_id="clean_dataset"
    )
    def clean_dataset(data_interval_start: Optional[DateTime]=None):
        """
        ### Clean Dataset
        Extracts data from the following Excel yearly-partitioned spreadsheets stored
        on the `bucket.raw_data.name` GCS bucket.  

        Processes the data to clean it
        & loads them as parquet files in into the `bucket.datasets.name` GCS bucket
        as year-partitioned parquet files under the `bucket.datasets.scores_prefix`.
        """
        raw_data, year = config["buckets"]["raw_data"], data_interval_start.year # type: ignore
        gcs = GCSHook()

        # Download data as Excel Spreadsheets
        # download Sec 4 Cohort sendout spreadsheet
        s4_path = f"{raw_data['sec4_prefix']}/{year}.xlsx"
        if not gcs.exists(raw_data["name"], s4_path):
            raise FileNotFoundError(f"Expected S4 Cohort Excel Spreadsheet to exist: {s4_path}")
        gcs.download(raw_data["name"], s4_path, f"s4_{year}.xlsx")
        # download P6 screening spreadsheet if it exists
        p6_path = f"{raw_data['p6_prefix']}/{year}.xlsx"
        if gcs.exists(raw_data["name"], p6_path):
            gcs.download(raw_data["name"], f"p6_{year}.xlsx")

        # Extract & Transform data from spreadsheets to Parquet
        df = prepare_extract(pd.read_excel(f"s4_{year}.xlsx"), year)
        # merge in p6 screening data if present
        if path.exists(f"p6_{year}.xlsx"):
            p6_df = prepare_p6(pd.read_excel(f"p6_{year}.xlsx"))
            df = pd.merge(df, p6_df, how="left", on="Serial number")
        # serial no. column no longer needed post join.
        df = df.drop(columns=["Serial number"])
        # write dataframe as parquet files
        df.to_parquet(f"{year}.parquet")

        # Upload data as parquet files
        datasets = config["buckets"]["datasets"]
        gcs.upload(datasets["name"], 
                   object_name=f"{datasets['scores_prefix']}/{year}.parquet",
                   filename=f"{year}.parquet")
    clean_dataset()
dag = pipeline()
