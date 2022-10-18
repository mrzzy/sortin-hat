#
# sortin-hat
# Pipeline
# Airflow DAG
#

import random
import string
import unittest.mock as mock
from os.path import dirname
from pathlib import Path
from typing import Any, Dict, Generator, cast

import pandas as pd
import pytest
from airflow.models.dagbag import DagBag
from airflow.models.dagrun import DagRun
from airflow.models.taskinstance import TaskInstance
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.tz import timezone

from dag import DAG_ID, DAG_START_DATE, TIMEZONE, load_dataset, local_year, pipeline_dag


def build_dagrun(conf: Dict) -> DagRun:
    """Build a DagRun to run Airflow Task instances with the given configuration."""
    return pipeline_dag.create_dagrun(
        state=DagRunState.RUNNING,
        execution_date=datetime(2021, 1, 1, tz=timezone(TIMEZONE)),
        data_interval=(
            datetime(2020, 1, 1, tz=timezone(TIMEZONE)),
            datetime(2020, 12, 31, tz=timezone(TIMEZONE)),
        ),
        start_date=DAG_START_DATE,
        run_type=DagRunType.MANUAL,
        conf=conf,
    )


def build_task_instance(dagrun: DagRun, task_id: str) -> TaskInstance:
    """Construct a Task Instance with a DAG run to test the Airflow Task with the given task_id."""
    task_instance = dagrun.get_task_instance(task_id)
    if task_instance is None:
        raise ValueError(f"No task task found with task_id: {task_id}")
    task_instance.task = pipeline_dag.get_task(task_id)
    return task_instance


@pytest.fixture
def test_bucket() -> Generator[str, None, None]:
    """Creates a temporary GCS Bucket to stored results for testing."""
    # random suffix added to bucket name to prevent collisions
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    bucket_name = f"sss-sortin-hat-test-dag-{suffix}"

    gcs = GCSHook()
    gcs.create_bucket(bucket_name)
    yield bucket_name
    gcs.delete_bucket(bucket_name)


@pytest.mark.unit
def test_load_dataset(dummy_data: Dict[str, Any]):
    with mock.patch(
        "pandas.read_parquet",
        return_value=pd.DataFrame(dummy_data),
    ) as read_parquet:
        bucket, prefix, year = "bucket", "prefix", 2016

        # check: paritions merged into one dataframe
        expect_df = pd.DataFrame(dummy_data)
        expect_df["Year"] = year

        assert (
            (
                load_dataset(
                    datasets_bucket=bucket,
                    dataset_prefix=prefix,
                    years=[year],
                )
                == expect_df
            )
            .all()
            .all()
        )

        # check: read_parquet called with correct args
        read_parquet.assert_any_call(
            f"gs://{bucket}/{prefix}/{year}.pq",
        )


@pytest.mark.unit
def test_pipeline_dag_import():
    dagbag = DagBag(Path(f"{dirname(__file__)}/../src"), include_examples=False)
    # check: dag imported without errors & registered
    assert dagbag.import_errors == {}
    assert DAG_ID in dagbag.dags


@pytest.mark.integration
def test_transform_dataset(dagrun: DagRun, test_bucket: str):
    dagrun = build_dagrun(
        {
            "datasets_bucket": test_bucket,
        }
    )
    task_instance = build_task_instance(dagrun, "transform_dataset")
    task_instance.run(ignore_ti_state=True)

    # check: task executed successfully
    assert task_instance.state == TaskInstanceState.SUCCESS
    # check: parquet file written to dataset bucket
    assert GCSHook().exists(
        test_bucket, f"dataset/{local_year(cast(DateTime, dagrun.execution_date))}.pq"
    )
