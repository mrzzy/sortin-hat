#
# sortin-hat
# Pipeline
# Airflow DAG
#

import unittest.mock as mock
from os.path import dirname
from pathlib import Path

from airflow.models.connection import Connection
from airflow.models.dagbag import DagBag

from dag import DAG_ID, pd_storage_opts


def test_pd_storage_ops(gcp_connection: Connection):
    with mock.patch(
        "airflow.models.connection.Connection.get_connection_from_secrets",
        return_value=gcp_connection,
    ):
        assert pd_storage_opts(str(gcp_connection.id)) == {
            "token": gcp_connection.extra_dejson[
                "extra__google_cloud_platform__key_path"
            ]
        }


def test_pipeline_dag_import():
    dagbag = DagBag(Path(f"{dirname(__file__)}/../src"), include_examples=False)
    # check: dag imported without errors & registered
    assert dagbag.import_errors == {} and DAG_ID in dagbag.dags
