#
# Sort'in Hat
# Tests
# Shared Fixtures
#


from itertools import cycle
from typing import Any, Dict

import numpy as np
import pytest
from airflow.models.connection import Connection
from pandas import DataFrame
from pandas.core.tools.datetimes import islice

from extract import (
    CARDING_LEVELS,
    COURSE_MAPPING,
    GENDER_MAPPING,
    HOUSING_MAPPING,
    PSLE_MAPPING,
    PSLE_SUBJECTS,
    SPORTS_LEVEL_MAPPING,
)

DUMMY_PREFIX = "dummy_"


@pytest.fixture
def dummy_data() -> Dict[str, Any]:
    """
    Returns dict of 5 dummy columns with 3 rows each, 2 categorical, 3 numeric.
    Useful for constructing dummy Pandas DataFrames.
    """
    data = {
        f"{DUMMY_PREFIX}cat_{i}": [f"{j}" for j in range(3)] for i in range(2)
    }  # type: Dict[str, Any]
    data.update({f"{DUMMY_PREFIX}num_{i}": np.arange(3) for i in range(3)})
    return data


@pytest.fixture
def gcp_connection() -> Connection:
    """Returns a fake GCP Airflow Connection for testing."""
    return Connection(
        conn_id="google_cloud_default",
        conn_type="google-cloud-platform",
        extra={"extra__google_cloud_platform__key_path": "key.json"},
    )


@pytest.fixture
def extract_df(dummy_data: Dict[str, Any]) -> DataFrame:
    """Returns a dataframes with dummy data & columns expected for feature extraction"""
    n_keys = lambda mapping, n: list(islice(cycle(mapping.keys()), n))
    test_data = {
        # UNKNOWN added to verify mapped default values
        "Sec4_CardingLevel": CARDING_LEVELS[:2] + ["UNKNOWN"],
        "Gender": n_keys(GENDER_MAPPING, 3),
        "Sec4_SportsLevel": n_keys(SPORTS_LEVEL_MAPPING, 2) + ["UNKNOWN"],
        "Course": n_keys(COURSE_MAPPING, 3),
        "ResidentialType": n_keys(HOUSING_MAPPING, 3),
        "Score": np.arange(3),
    }
    test_data.update({subject: n_keys(PSLE_MAPPING, 3) for subject in PSLE_SUBJECTS})
    test_data.update(dummy_data)

    return DataFrame(test_data)
