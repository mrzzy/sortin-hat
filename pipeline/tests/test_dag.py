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

from dag import DAG_ID


def test_pipeline_dag_import():
    with mock.patch(
        "airflow.models.connection.Connection.get_connection_from_secrets",
        Connection(
            conn_id="google_cloud_default",
            conn_type="google-cloud-platform",
            extra={"extra__google_cloud_platform__key_path": "key.json"},
        ),
    ):

        dagbag = DagBag(Path(f"{dirname(__file__)}/../src"), include_examples=False)
        # check: dag imported without errors & registered
        assert dagbag.import_errors == {} and DAG_ID in dagbag.dags
