#
# Sort'in Hat
# Models
# Clean Data Unit Tests
#

import pandas as pd

from clean import P6_COLUMNS, clean_p6


def test_clean_p6():
    # test values designed to test cleaning operations
    n_rows = 3
    test_values = {
        "Unnamed: 0": ["1", "2", "x"],
        "Q1 M": ["3", "not a number", "4"],
        "unused": list(range(n_rows)),
    }
    # cleaning operations do not apply to other columns, add dummy values for them
    dummy_cols = set(P6_COLUMNS).difference(test_values.keys())
    serial_no = "Serial number"
    test_values.update(
        {col: list(range(n_rows)) for col in dummy_cols if col != serial_no}
    )

    df = clean_p6(pd.DataFrame(test_values))

    # check: discards non-selected columns
    assert not "unused" in df.columns
    # check: all selected columns are present
    assert all([c in df.columns for c in P6_COLUMNS])
    # check: renamed from "Unnamed: 0", dropped invalid number 'x'
    assert (df[serial_no] == pd.Series([1, 2], name=serial_no)).all()
    # check: 'Q1 M' column values coerced into numeric
    assert df.loc[df[serial_no] == 2, "Q1 M"].isna().all()
