#
# sortin-hat
# Training Objective
# Unit Tests
#

import unittest.mock as mock
from typing import Any, Dict
from unittest.mock import Mock, call

import pandas as pd
import pytest

from train import load_dataset, run_objective


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
# ignore pandas slice assignment & ray convergence warnings
@pytest.mark.filterwarnings("ignore")
def test_run_objective(extract_df: pd.DataFrame):
    # check: exception on missing gcp credentials
    with pytest.raises(LookupError):
        run_objective({})

    @mock.patch("ray.tune.report")
    @mock.patch("pandas.read_parquet", return_value=extract_df)
    def test(pd_read_parquet: Mock, ray_tune_report: Mock):
        bucket, prefix = "bucket", "prefix"
        with mock.patch.dict(
            "os.environ", values={"GOOGLE_APPLICATION_CREDENTIALS": "key.json"}
        ):
            run_objective(
                params={
                    "datasets_bucket": bucket,
                    "dataset_prefix": prefix,
                    "begin_year": 2016,
                    "current_year": 2021,
                    "model_name": "Linear Regression",
                    "l1_regularization": 1e-3,
                    "l2_regularization": 1e-3,
                }
            )

        # check: arguments passed to read read_parquet
        assert pd_read_parquet.call_args_list == [
            call(f"gs://{bucket}/{prefix}/{year}.pq") for year in range(2016, 2021 + 1)
        ]

        # check: metrics reported to ray tune
        assert (
            ray_tune_report.call_count == 1
            and ray_tune_report.call_args[1].keys()
            - {
                f"{subset}{metric}"
                for subset in ["train", "validate", "test"]
                for metric in ["r2", "mse", "rmse"]
            }
            == frozenset()
        )

    test()
