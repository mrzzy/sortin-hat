#
# sortin-hat
# Pipeline
# Airflow DAG
#

from typing import cast

from airflow.decorators import dag, task
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone

from clean import clean_extract, clean_p6
from transform import suffix_subject_level

TIMEZONE = timezone("Asia/Singapore")


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
    - Optional P6 Screening Template, stored as `raw_p6_prefix/<YEAR>.xlsx`

    The pipeline assumes that all dates / times are expressed in the Asia/Singapore time zone.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the
    `models_bucket` GCS bucket.
    """

    @task(
        task_id="transform_dataset",
    )
    def transform_dataset(data_interval_start: DateTime = cast(DateTime, None)):
        """
        Transform the Data source Excel Spreadsheets into Parquet Dataset.
        Both data source & dataset are partitioned by cohort year.
        """
        import pandas as pd
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        # load & Clean data from excel spreadsheet(s)
        year = data_interval_start.year
        df = clean_extract(
            pd.read_excel(f"gs://{raw_bucket}/{raw_s4_prefix}/{year}.xlsx")
        )
        # merge in cleaned p6 data if it exists
        gcs = GCSHook()
        if gcs.exists(raw_bucket, f"{raw_p6_prefix}/{year}.xlsx"):
            p6_df = clean_p6(pd.read_excel(f"gs://{raw_bucket}/{raw_p6_prefix}"))
            df = pd.merge(df, p6_df, how="left", on="Serial number")

        # suffix subjects with level the subject was taken
        df = suffix_subject_level(df, year)

        # write transformed dataset as compressed parquet file
        df.to_parquet(f"gs://{datasets_bucket}/dataset_{year}.pq")

    transform_dataset()


dag = pipeline()
