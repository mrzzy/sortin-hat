#
# Sort'in Hat
# Models
# Clean Data Unit Tests
#

import numpy as np
import pandas as pd

from clean import EXTRACT_DTYPE_OVERRIDES, P6_COLUMNS, clean_extract, clean_p6
from extract import PSLE_SUBJECTS

SERIAL_NO = "Serial number"


def test_clean_p6():
    # test values designed to test cleaning operations
    n_rows = 4
    test_values = {
        "Unnamed: 0": ["1", "2", "x", "3"],
        "Q1 M": ["3", "4", "5", "not a number"],
        "unused": list(range(n_rows)),
    }
    # cleaning operations do not apply to other columns, add dummy values for them
    dummy_cols = set(P6_COLUMNS).difference(test_values.keys())
    test_values.update(
        {col: list(range(n_rows)) for col in dummy_cols if col != SERIAL_NO}
    )

    df = clean_p6(pd.DataFrame(test_values))

    # check: discards non-selected columns
    assert not "unused" in df.columns
    # check: serial no. renamed from "Unnamed: 0", dropped invalid number 'x'
    assert (df[SERIAL_NO] == pd.Series([1, 2], name=SERIAL_NO)).all()
    # check: all selected columns are present
    assert all([c in df.columns for c in P6_COLUMNS])
    # check: dropped string in "Q1 M" column
    assert (df["Q1 M"] == pd.Series([3, 4], name="Q1 M")).all()


def test_clean_extract():
    data = {
        "missing": ["-", "0", 0],
        "Serial number": [1, 2.0, 3],
        "Sec4_BoardingStatus": np.arange(1, 4),
        "Sec4_SportsLevel": ["L1", "1", np.nan],
        "Course": ["Express", 2, "Normal"],
    }
    data.update({subject: np.arange(3) for subject in PSLE_SUBJECTS})
    df = clean_extract(pd.DataFrame(data))

    # check: missing values are converted to nan
    assert df["missing"].isna().all()
    # check: dtypes overridden for specific columns
    expected = [
        (column, np.object_ if dtype == np.str_ else dtype)
        for column, dtype in EXTRACT_DTYPE_OVERRIDES.items()
    ]
    for column, dtype in expected:
        assert df[column].dtype == dtype

    # check: 'Sec4_BoardingStatus' renamed to 'BoardingStatus'
    assert (df["BoardingStatus"] == np.arange(1, 4)).all()
    # check: leading 'L' in 'Sec4_SportsLevel' stripped
    assert (df["Sec4_SportsLevel"][:2] == np.array(["1", "1"])).all() and np.isnan(
        df["Sec4_SportsLevel"][2]
    )
