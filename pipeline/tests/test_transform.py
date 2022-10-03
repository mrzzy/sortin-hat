# Sort'in Hat
# Models
# Transform Data Unit Tests
#

from typing import Any, Dict

import numpy as np
import pandas as pd
from conftest import FORWARDED_PREFIX

from transform import suffix_subject_level

SUBJECTS = ["English", "Maths", "Science"]


def test_suffix_subject_level(forwarded_data: Dict[str, Any]):
    n_rows = 3
    # test columns to verify forwarding of non subject columns
    test_data = forwarded_data
    # generate test data for cohort year spanning back n_years
    cohort_year, n_years = 2016, 4
    for i in range(1, n_years):
        current_year = cohort_year - i
        # current year result divider column
        test_data[f"{current_year} Results"] = np.empty(n_rows)
        # current year results for each subject
        test_data.update(
            {
                f"{subject}.{i+1}": np.arange(n_rows, dtype=np.float_)
                for subject in SUBJECTS
            }
        )

    df = suffix_subject_level(pd.DataFrame(test_data), cohort_year)

    # check: check test columns are forwarded correctly
    forwarded_cols = frozenset(
        [forwarded for forwarded in test_data.keys() if FORWARDED_PREFIX in forwarded]
    )
    assert all([f in df.columns for f in forwarded_cols])

    # group columns by result year divider column
    subject_df = (
        df.drop(columns=forwarded_cols).columns.to_frame().rename(columns={0: "Year"})
    )
    # forward fill which result year each subject column belongs to
    subject_df.loc[~subject_df["Year"].str.contains("Results"), :] = pd.NA  # type: ignore
    subject_df = subject_df.ffill()
    subject_df = subject_df[~subject_df.index.str.contains("Results")]

    for year_col, subject_cols in subject_df.groupby("Year"):
        # check: subjects are tagged with expected secondary school level
        year = int(year_col.split()[0])
        level = n_years - (cohort_year - year)
        assert all(
            [f"{subject} [S{level}]" in subject_cols.index for subject in SUBJECTS]
        )
