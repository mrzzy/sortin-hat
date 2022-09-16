#
# sortin-hat
# Data & ML Pipeline
# Airflow DAG
#

from datetime import timedelta
from typing import List

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from pendulum import date

config = {
    "buckets": {
        "raw_data": "sss-sortin-hat-raw-data",
        "dataset": "sss-sortin-hat-datasets",
    }
}

@dag(
    dag_id="sortin-hat-pipeline",
    description="Data & ML Pipeline producing Sortin-hat ML models",

    # each dag run handles a year-sized data interval from start_date
    start_date=date(2016, 1, 1),
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
    - Sec 4 Cohort Sendout: stored under 'Sec4_Cohort/<YEAR>.xlsx'
    - P6 Screening Template: stored under 'P6_Screening/<YEAR>.xlsx'

    The Excel spreadsheet should be partitioned by year & stored in the 'raw_data'
    GCS Bucket.

    ## Outputs
    MLFlow Models & Evaluation results from the ML training process stored in the 
    'models' directory.

    """
    @task(
        task_id="etl_dataset"
        description
    )
    def etl_dataset(objects: List[str]):
        """
        ### Extract, Transform & Load (ETL) Excel into Parquet files
        Extracts data from the following Excel yearly-partitioned spreadsheets stored
        on the 'raw_data' GCS bucket.

        Performs transformation to clean the data & exports them as parquet files
        in then 'dataset' GCS bucket.
        """
