#
# sortin-hat
# Pipeline
# Airflow DAG
#

from os import path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, cast

import pandas as pd
from airflow.decorators import dag, task
from airflow.models.dag import DAG
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone

from clean import clean_p6
from prepare import prepare_extract

TIMEZONE = timezone("Asia/Singapore")
config = {
    "buckets": {
        "raw_data": {
            "name": "sss-sortin-hat-raw-data",
            "sec4_prefix": "Sec4_Cohort",
            "p6_prefix": "P6_Screening",
        },
        "datasets": {
            "name": "sss-sortin-hat-datasets",
            "clean_prefix": "clean",
        },
    }
}


@dag(
    dag_id="sortin-hat-pipeline",
    description="Sortin-hat ML Pipeline",
    # each dag run handles a year-sized data interval from start_date
    start_date=datetime(2016, 1, 1, tz=TIMEZONE),
    schedule_interval="@yearly",
)
def pipeline(
    raw_bucket: str = "sss-sortin-hat-raw-data",
    raw_s4_prefix: str = "Sec4_Cohort",
    raw_p6_prefix: str = "P6_Screening",
    datasets_bucket: str = "sss-sortin-hat-datasets",
    models_bucket: str = "sss-sortin-hat-models",
):
    """
    # Sortin-hat ML Pipeline
    End to End Pipeline for training Sortin-hat ML models for predicting student scores.

    ## Prerequisites
    ### Connections
    Expects a GCP connection to be configured with the id `google_cloud_default`

    ### Infrastructure
    GCS buckets `raw_bucket`, `datasets_bucket` & `models_bucket` should be created
    beforehand.

    ### Data Source
    The pipeline takes in as data source 2 kinds of Excel Spreadsheets stored
    in the `raw_bucket` GCS bucket, partitioned by year:
    - Sec 4 Cohort Sendout, stored as `raw_s4_prefix/<YEAR>.xlsx`
    - P6 Screening Template, stored as `raw_p6_prefix/<YEAR>.xlsx`

    The pipeline assumes that all dates / times are expressed in the Asia/Singapore time zone.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the
    `models_bucket` GCS bucket.
    """

    @task(task_id="clean_data")
    def clean_data(data_interval_start: Optional[DateTime] = None):
        """
        ### Clean Data
        Extracts data from the following Excel yearly-partitioned spreadsheets stored
        on the `bucket.raw_data.name` GCS bucket.

        Processes the data to clean it
        & loads them as parquet files in into the `bucket.datasets.name` GCS bucket
        as year-partitioned parquet files under the `bucket.datasets.clean_prefix`.
        """
        sg_begin = cast(DateTime, data_interval_start).astimezone(TIMEZONE)
        raw_data, year = config["buckets"]["raw_data"], sg_begin.year
        gcs = GCSHook()

        # Download data as Excel Spreadsheets
        with TemporaryDirectory(prefix=str(year)) as work_dir:
            # download Sec 4 Cohort sendout spreadsheet
            src_s4_path = f"{raw_data['sec4_prefix']}/{year}.xlsx"
            dest_s4_path = path.join(work_dir, "s4.xlsx")
            if not gcs.exists(raw_data["name"], src_s4_path):
                raise FileNotFoundError(
                    f"Expected S4 Cohort Excel Spreadsheet to exist: {src_s4_path}"
                )
            gcs.download(raw_data["name"], src_s4_path, dest_s4_path)
            # download P6 screening spreadsheet if it exists
            src_p6_path = f"{raw_data['p6_prefix']}/{year}.xlsx"
            dest_p6_path = path.join(work_dir, "p6.xlsx")
            if gcs.exists(raw_data["name"], src_p6_path):
                gcs.download(raw_data["name"], src_p6_path, dest_p6_path)

            # Extract & Transform data from spreadsheets to Parquet
            df = prepare_extract(pd.read_excel(dest_s4_path), year)
            # merge in p6 screening data if present
            if path.exists(dest_p6_path):
                # header=1: headers are stored in p6 data on the second row
                p6_df = clean_p6(pd.read_excel(dest_p6_path, header=1))
                df = pd.merge(df, p6_df, how="left", on="Serial number")

            # serial no. column no longer needed post join.
            df = df.drop(columns=["Serial number"])
            # write dataframe as parquet files
            df.to_parquet(f"{year}.parquet")

            # Upload data as parquet files
            datasets = config["buckets"]["datasets"]
            gcs.upload(
                datasets["name"],
                object_name=f"{datasets['clean_prefix']}/{year}.parquet",
                filename=f"{year}.parquet",
            )

    clean_data()
