#
# sortin-hat
# Pipeline
# Airflow DAG
#

import json
from typing import Dict, cast

from airflow.decorators import dag, task
from airflow.models.connection import Connection
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone

from clean import clean_extract, clean_p6
from extract import extract_features
from transform import suffix_subject_level

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
    return pd.concat(
        [
            pd.read_parquet(
                f"gs://{datasets_bucket}/{dataset_prefix}/{year}.pq",
                storage_options=pd_storage_opts(gcp_connection_id),
            )
            for year in years
        ]
    )


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
    # extract gcp service account json key path from airflow gcp connection

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
        data_interval_end: DateTime = cast(DateTime, None),
    ) -> str:
        """
        Transform the Data source Excel Spreadsheets into Parquet Dataset.
        Both data source & dataset are partitioned by cohort year.
        Returns the path the Parquet Dataset in GCS.
        """
        import pandas as pd
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        # data interval's year in local time zone
        year = data_interval_end.astimezone(timezone(timezone_str)).year

        # load & Clean data from excel spreadsheet(s)
        storage_opts = pd_storage_opts(gcp_connection_id)
        df = clean_extract(
            pd.read_excel(
                f"gs://{raw_bucket}/{raw_s4_prefix}/{year}.xlsx",
                storage_options=storage_opts,
            )
        )

        # suffix subjects columns with level the subject was taken
        df = suffix_subject_level(df, year)

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
        dataset_path = f"gs://{datasets_bucket}/dataset_{year}.pq"
        df.to_parquet(dataset_path, storage_options=storage_opts)

        return dataset_path

    @task(task_id="train_model")
    def train_model(
        gcp_connection_id: str,
        datasets_bucket: str,
        timezone_str: str,
        start_date: DateTime = cast(DateTime, None),
        data_interval_end: DateTime = cast(DateTime, None),
    ):
        """
        Train as a Machine Learning model.
        """
        import pandas as pd
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        # convert timestamps into the local timezone
        local_tz = timezone(timezone_str)
        start_date = start_date.astimezone(local_tz)
        data_interval_end = data_interval_end.astimezone(local_tz)

        # read all dataset partitions from DAG's start date till the end of the data interval
        gcs, storage_opts = GCSHook(), pd_storage_opts(gcp_connection_id)
        partition_paths = [
            f"gs://{datasets_bucket}/dataset_{year}.pq"
            for year in range(start_date.year, data_interval_end.year)
            if gcs.exists(datasets_bucket, f"dataset_{year}.pq")
        ]
        df = pd.concat(
            [pd.read_parquet(p, storage_options=storage_opts) for p in partition_paths]
        )

    transform_dataset(
        gcp_connection_id,
        raw_bucket,
        raw_s4_prefix,
        raw_p6_prefix,
        datasets_bucket,
        timezone_str,
    )


dag = pipeline()
