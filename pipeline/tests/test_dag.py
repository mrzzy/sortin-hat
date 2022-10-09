#
# sortin-hat
# Pipeline
# Airflow DAG
#

import unittest.mock as mock
from os.path import dirname
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from airflow.models.connection import Connection
from airflow.models.dagbag import DagBag

from dag import DAG_ID, load_dataset, pd_storage_opts


def test_pd_storage_opts(gcp_connection: Connection):
    with mock.patch(
        "airflow.models.connection.Connection.get_connection_from_secrets",
        return_value=gcp_connection,
    ):
        # check: storage opts contains path to json key
        assert pd_storage_opts(str(gcp_connection.id)) == {
            "token": gcp_connection.extra_dejson[
                "extra__google_cloud_platform__key_path"
            ]
        }


def test_load_dataset(dummy_data: Dict[str, Any], gcp_connection: Connection):
    with mock.patch(
        "airflow.models.connection.Connection.get_connection_from_secrets",
        return_value=gcp_connection,
    ), mock.patch(
        "pandas.read_parquet",
        return_value=pd.DataFrame(dummy_data),
    ) as read_parquet:
        bucket, prefix, years = "bucket", "prefix", [2016, 2017]
        gcp_id = str(gcp_connection.id)

        # check: paritions merged into one dataframe
        assert (
            (
                load_dataset(
                    gcp_id,
                    datasets_bucket=bucket,
                    dataset_prefix=prefix,
                    years=years,
                )
                == pd.concat([pd.DataFrame(dummy_data) for _ in range(2)])
            )
            .all()
            .all()
        )

        # check: read_parquet called with correct args
        for year in years:
            read_parquet.assert_any_call(
                f"gs://{bucket}/{prefix}/{year}.pq",
                storage_options=pd_storage_opts(gcp_id),
            )


def test_pipeline_dag_import():
    dagbag = DagBag(Path(f"{dirname(__file__)}/../src"), include_examples=False)
    # check: dag imported without errors & registered
    assert dagbag.import_errors == {} and DAG_ID in dagbag.dags
