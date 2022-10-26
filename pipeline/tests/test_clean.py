#
# Sort'in Hat
# Models
# Clean Data Unit Tests
#

from typing import Type, Union

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from clean import EXTRACT_SCHEMA, P6_COLUMNS, SERIAL_NO, clean_extract, clean_p6
from extract import PSLE_MAPPING, PSLE_SUBJECTS


@pytest.mark.unit
def test_clean_p6():
    # test values designed to test cleaning operations
    n_rows = 4
    test_values = {
        "Unnamed: 0": ["1", "2", "x", "3"],
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
    assert (df[SERIAL_NO] == np.array([1, 2, 3])).all()
    # check: all selected columns are present
    assert all([c in df.columns for c in expected_cols])
    # check: data type schema is enforced
    assert (df[P6_COLUMNS].dtypes == np.float_).all() and df[SERIAL_NO].dtype == np.int_


@pytest.mark.unit
def test_clean_extract():
    n_rows = 6
    data = {
        "missing": ["-", "0", 0, "-", "0", "-"],
        SERIAL_NO: [1, 2, 3, np.nan, 5, 6],
        "Sec4_BoardingStatus": list(range(1, n_rows + 1)),
        "Sec4_SportsLevel": ["L1", "1", "", "L2", "L1", "1"],
    }
    data.update(
        {subject: list(PSLE_MAPPING.keys())[:n_rows] for subject in PSLE_SUBJECTS}
    )

    # add dummy data to columns expected in extract schema
    def test_data(dtype: Union[Type[np.str_], Type[np.float_]]) -> NDArray:
        if dtype == np.str_:
            return np.array(["DUMMY" for _ in range(n_rows)])
        if dtype == np.float_:
            return np.arange(n_rows)
        raise ValueError(f"Unsupported dtype: {dtype}")

    data.update(
        {
            column: test_data(dtype)
            for column, dtype in EXTRACT_SCHEMA.items()
            if column not in data
        }
    )

    df = clean_extract(pd.DataFrame(data))
    # check: missing values are converted to nan
    assert df["missing"].isna().all()
    # check: data type schema is enforced
    expect_schema = {
        column: np.object_ if dtype == np.str_ else dtype
        for column, dtype in EXTRACT_SCHEMA.items()
    }
    assert (df[PSLE_SUBJECTS].dtypes == np.object_).all() and (
        df[expect_schema.keys()].dtypes == pd.Series(expect_schema)
    ).all()

    # check: missing serial no. dropped
    assert (df[SERIAL_NO] == np.array([1, 2, 3, 5, 6])).all()

    # check: leading 'L' in 'Sec4_SportsLevel' stripped
    assert (df["Sec4_SportsLevel"] == np.array(["1", "1", "", "1", "1"])).all()
    # check: 'Sec4_BoardingStatus' renamed to 'BoardingStatus'
    assert not "Sec4_BoardingStatus" in df and "BoardingStatus" in df
