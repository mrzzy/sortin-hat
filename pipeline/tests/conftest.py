#
# Sort'in Hat
# Tests
# Shared Fixtures
#


from typing import Any, Dict

import numpy as np
import pytest

DUMMY_PREFIX = "forwarded_"


@pytest.fixture
def dummy_data() -> Dict[str, Any]:
    """
    Returns dict of 5 dummy columns with 3 rows each useful for constructing
    dummy Pandas DataFrames.
    """
    return {f"{DUMMY_PREFIX}{i}": np.arange(3, dtype=np.float_) for i in range(5)}
