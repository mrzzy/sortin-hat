#
# Sort'in Hat
# Tests
# Shared Fixtures
#


from typing import Any, Dict

import numpy as np
import pytest

FORWARDED_PREFIX = "forwarded_"


@pytest.fixture
def forwarded_data() -> Dict[str, Any]:
    """
    Returns dict of 5 dummy columns with 3 rows each to verify forwarding.
    """
    return {f"{FORWARDED_PREFIX}{i}": np.arange(3, dtype=np.float_) for i in range(5)}
