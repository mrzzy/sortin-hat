#
# Sort'in Hat
# Tests
# Shared Fixtures
#


from typing import Any, Dict

import numpy as np
import pytest
from airflow.models.connection import Connection

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
