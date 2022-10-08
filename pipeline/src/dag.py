#
# sortin-hat
# Pipeline
# Airflow DAG
#

import json
from typing import cast

from airflow.decorators import dag, task
from airflow.models.connection import Connection
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone

from clean import clean_extract, clean_p6
from transform import suffix_subject_level

TIMEZONE = "Asia/Singapore"
DAG_ID = "sortin-hat-pipeline"


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

    The pipeline assumes that all dates / times are expressed in the Asia/Singapore time zone.

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
        timezone_str: str,
        data_interval_start: DateTime = cast(DateTime, None),
    ) -> str:
        """
        Transform the Data source Excel Spreadsheets into Parquet Dataset.
        Both data source & dataset are partitioned by cohort year.
        Returns the path the Parquet Dataset in GCS.
        """
        print("raw_bucket:", raw_bucket)
        print("raw_s4_prefix:", raw_s4_prefix)
        print("raw_p6_prefix:", raw_p6_prefix)
        print("datasets_bucket:", datasets_bucket)
        print("data_interval_start", data_interval_start)
        import pandas as pd
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        # data interval's year in local time zone
        year = data_interval_start.astimezone(timezone(timezone_str)).year

        # extract gcp service account json key path from airflow gcp connection
        gcp_key_path = json.loads(
            Connection.get_connection_from_secrets(gcp_connection_id).get_extra()  # type: ignore
        )["extra__google_cloud_platform__key_path"]

        # load & Clean data from excel spreadsheet(s)
        storage_options = {"token": gcp_key_path}
        df = clean_extract(
            pd.read_excel(
                f"gs://{raw_bucket}/{raw_s4_prefix}/{year}.xlsx",
                storage_options=storage_options,
            )
        )

        # suffix subjects columns with level the subject was taken
        df = suffix_subject_level(df, year)

        # merge in cleaned p6 data if it exists
        gcs = GCSHook()
        if gcs.exists(raw_bucket, f"{raw_p6_prefix}/{year}.xlsx"):
            p6_df = clean_p6(
                pd.read_excel(
                    # header=1: headers are stored in p6 data on the second row
                    f"gs://{raw_bucket}/{raw_p6_prefix}/{year}.xlsx",
                    storage_options=storage_options,
                    header=1,
                )
            )
            df = pd.merge(df, p6_df, how="left", on="Serial number")

        # write transformed dataset as compressed parquet file
        dataset_path = f"gs://{datasets_bucket}/dataset_{year}.pq"
        df.to_parquet(dataset_path, storage_options=storage_options)

        return dataset_path

    transform_dataset(
        gcp_connection_id,
        raw_bucket,
        raw_s4_prefix,
        raw_p6_prefix,
        datasets_bucket,
        timezone_str,
    )


dag = pipeline()
