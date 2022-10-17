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
import pytest
from airflow.models.dagbag import DagBag

from dag import DAG_ID, load_dataset


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
