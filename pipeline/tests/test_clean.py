#
# Sort'in Hat
# Models
# Clean Data Unit Tests
#

import numpy as np
import pandas as pd
import pytest

from clean import P6_COLUMNS, clean_extract, clean_p6
from extract import PSLE_MAPPING, PSLE_SUBJECTS

SERIAL_NO = "Serial number"


@pytest.mark.unit
def test_clean_p6():
    # test values designed to test cleaning operations
    n_rows = 4
    test_values = {
        "Unnamed: 0": ["1", "2", "x", "3"],
        "Q1 M": ["3", "4", "5", "not a number"],
        "unused": list(range(n_rows)),
    }
    # cleaning operations do not apply to other columns, add dummy values for them
    expected_cols = set(P6_COLUMNS + [SERIAL_NO])
    dummy_cols = expected_cols.difference(test_values.keys())
    test_values.update(
        {col: list(range(n_rows)) for col in dummy_cols if col != SERIAL_NO}
    )

    df = clean_p6(pd.DataFrame(test_values))

    # check: discards non-selected columns
    assert not "unused" in df.columns
    # check: serial no. renamed from "Unnamed: 0", dropped invalid number 'x'
    assert (df[SERIAL_NO] == pd.Series([1, 2], name=SERIAL_NO)).all()
    # check: all selected columns are present
    assert all([c in df.columns for c in expected_cols])
    # check: dropped string in "Q1 M" column
    assert (df["Q1 M"] == pd.Series([3, 4], name="Q1 M")).all()


@pytest.mark.unit
def test_clean_extract():
    data = {
        "missing": ["-", "0", 0, "-", "0", "-"],
        "Serial number": [1, 2, 3, np.nan, 5, 6],
        "Sec4_BoardingStatus": list(range(1, 7)),
        "Sec4_SportsLevel": ["L1", "1", np.nan, "L2", "L1", "1"],
        "Course": ["Express", "Express", "Normal", "Normal", 1, " "],
    }
    data.update({subject: list(PSLE_MAPPING.keys())[:6] for subject in PSLE_SUBJECTS})
    df = clean_extract(pd.DataFrame(data))

    # check: missing values are converted to nan
    assert df["missing"].isna().all()
    # check: dtypes overridden for specific columns
    assert df["Sec4_SportsLevel"].dtype == np.object_
    assert all([dtype == np.object_ for dtype in df[PSLE_SUBJECTS].dtypes])

    # check: missing serial no. dropped
    assert (df["Serial number"] == np.array([1, 2, 3])).all()
    # check: bad course dropped.
    assert (df["Course"] == np.array(["Express", "Express", "Normal"])).all()

    # check: leading 'L' in 'Sec4_SportsLevel' stripped
    print(df["Sec4_SportsLevel"])
    assert (df["Sec4_SportsLevel"] == np.array(["1", "1", ""])).all()
    # check: 'Sec4_BoardingStatus' renamed to 'BoardingStatus'
    assert not "Sec4_BoardingStatus" in df and "BoardingStatus" in df
