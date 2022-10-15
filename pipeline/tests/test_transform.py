#
# Sort'in Hat
# Models
# Transform Data Unit Tests
#

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from transform import suffix_subject_level, unpivot_subjects

# generate test data for cohort year spanning back n_years
SUBJECTS = ["English", "Maths", "Science"]
COHORT_YEAR = 2016
N_YEARS = 4


@pytest.fixture
def subject_df(dummy_data: Dict[str, Any]) -> pd.DataFrame:
    n_rows = 3
    # test columns to verify forwarding of non subject columns
    test_data = dummy_data.copy()
    for i in range(N_YEARS):
        current_year = COHORT_YEAR - i
        # current year result divider column
        test_data[f"{current_year} Results"] = np.empty(n_rows)
        # current year results for each subject
        test_data.update(
            {
                f"{subject}.{i+1}": np.arange(n_rows, dtype=np.float_)
                for subject in SUBJECTS
            }
        )

    return pd.DataFrame(test_data)


@pytest.mark.unit
def test_suffix_subject_level(subject_df: pd.DataFrame, dummy_data: Dict[str, Any]):
    df = suffix_subject_level(subject_df, COHORT_YEAR)

    # check: check dumy columns are forwarded correctly
    assert all([f in df.columns for f in dummy_data.keys()])

    # group columns by result year divider column
    subject_df = (
        df.drop(columns=dummy_data.keys())
        .columns.to_frame()
        .rename(columns={0: "Year"})
    )
    # forward fill which result year each subject column belongs to
    subject_df.loc[~subject_df["Year"].str.contains("Results"), :] = pd.NA  # type: ignore
    subject_df = subject_df.ffill()
    subject_df = subject_df[~subject_df.index.str.contains("Results")]

    for year_col, subject_cols in subject_df.groupby("Year"):
        # check: subjects are tagged with expected secondary school level
        year = int(year_col.split()[0])
        level = N_YEARS - (COHORT_YEAR - year)
        assert all(
            [f"{subject} [S{level}]" in subject_cols.index for subject in SUBJECTS]
        )


@pytest.mark.unit
def test_unpivot_subjects(subject_df: pd.DataFrame, dummy_data: Dict[str, Any]):
    subject_cols = [column for column in subject_df.columns if "." in column]
    # suffix subject levels to test data
    subject_renames = {
        subject: "{0} [S{1}]".format(*subject.split(".")) for subject in subject_cols
    }
    subject_df = subject_df.rename(columns=subject_renames)

    df = unpivot_subjects(subject_df, COHORT_YEAR)

    # check: check dummy columns are forwarded correctly
    assert all([f in df.columns for f in dummy_data.keys()])
    # check: subject columns has been unpivoted
    actual_subjects = df["Subject"].unique()
    assert all([subject in subject_renames.values() for subject in actual_subjects])
    # check: missing scores dropped
    assert not df["Score"].isna().any()
    # check: secondary level extracted as "Level" column
    assert (df["Level"] == df["Subject"].str.extract(r"\[S(\d)\]", expand=False)).all()
