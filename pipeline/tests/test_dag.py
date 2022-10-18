#
# sortin-hat
# Pipeline Airflow DAG
# Unit Tests
#

import json
import os
import random
import string
import time
from os.path import abspath, dirname
from pathlib import Path
from typing import Callable, Generator

import pytest
from airflow.models.dagbag import DagBag
from google.cloud import storage
from pendulum import datetime
from pendulum.tz.timezone import Timezone
from testcontainers.compose import DockerCompose

from dag import DAG_ID, TIMEZONE

# resolve path to the project root directory
PROJECT_ROOT = abspath(f"{dirname(__file__)}/../..")


def load_dotenv(path: str):
    """Load the env vars defined in the .env file at the given path."""
    with open(path, "r") as f:
        for name, value in [
            line.strip().split("=", maxsplit=1) for line in f.readlines()
        ]:
            os.environ[name] = value


# Unit Tests
@pytest.mark.unit
def test_pipeline_dag_import():
    dagbag = DagBag(Path(f"{dirname(__file__)}/../src"), include_examples=False)
    # check: dag imported without errors & registered
    assert dagbag.import_errors == {}
    assert DAG_ID in dagbag.dags


# Integration Tests
@pytest.fixture
def test_bucket() -> Generator[str, None, None]:
    """Creates a temporary GCS Bucket to stored results for testing."""
    # random suffix added to bucket name to prevent collisions
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    bucket_name = f"sss-sortin-hat-test-dag-{suffix}"

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    gcs = storage.Client()
    bucket = gcs.create_bucket(bucket_name)
    yield bucket_name
    bucket.delete()


def poll(call: Callable[[], bool], timeout_secs: int = 30) -> bool:
    """Repeated call the given callback until its succeeeds or a timeout occurs.

    Args:
        call:
            Callable to repeated call. It should return True to to signal success,
            False otherwise.
        timeout_secs:
            No. of seconds before a timeout occurs.
    Returns:
        True if the call succeeded before the timeout, False otherwise
    """
    # monotonic clock is used to guard against clock skew
    begin = time.monotonic()
    while time.monotonic() - begin < timeout_secs:
        # exit immediately if the call succeeds
        if call():
            return True
    return False


@pytest.mark.integration
def test_pipeline_dag(test_bucket: str):
    with DockerCompose(
        PROJECT_ROOT,
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        pull=True,
        build=True,
    ) as c:
        # wait for airflow to start listening for connections
        airflow_address = "http://localhost:8080"
        c.wait_for(airflow_address)

        # check: pipeline dag run executes successfully
        params = {
            "datasets_bucket": test_bucket,
            "models_bucket": test_bucket,
        }
        logical_date = datetime(2021, 1, 1, tz=Timezone(TIMEZONE))
        assert (
            c.exec_in_container(
                "airflow",
                f"airflow dag test -t '{json.dumps(params)}' sortin-hat-pipeline"
                f"transform_dataset {logical_date.strftime('%Y-%m-%d')}",
            )
            == 0
        )

    # check: pipeline dag prdu


#    assert GCSHook().exists(
#
#        test_bucket, f"dataset/{local_year(cast(DateTime, dagrun.execution_date))}.pq
#    )
