#
# Sort'in Hat
# Models
# Feature Extraction
#

from typing import Any, Dict

import numpy as np
import pandas as pd

from extract import PSLE_SUBJECTS, encode_psle


def test_encode_psle(forwarded_data: Dict[str, Any]):
    # check: columns forwarded unaltered
    grades = ["A*", "A", "B", "C", "D", "E", "F"]
    test_values = {subject: grades[:3] for subject in PSLE_SUBJECTS}
    test_values.update(forwarded_data)
    df = encode_psle(pd.DataFrame(test_values))
    assert all(
        [(df[key] == forwarded_data[key]).all() for key in forwarded_data.keys()]
    )

    # check: psle subjects encoded
    df = encode_psle(pd.DataFrame({subject: grades for subject in PSLE_SUBJECTS}))
    assert all(
        [(df[subject] == np.array(range(1, 8))).all() for subject in PSLE_SUBJECTS]
    )
