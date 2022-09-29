#
# Sort'in Hat
# Models
# Transform Data
#

import re

import pandas as pd


def suffix_subject_level(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Suffix subject columns in the given 'Data Extraction' dataframe with the
    secondary school level the subject was taken.

    For example: English columns will be suffixed: "English [S1]", "English [S2]"
    to signify that the subject was taken in Secondary 1 & 2 respectively.

    Infers the level the subject taken from the 'YYYY Results' columns in the
    Dataframe.
    Args:
        df:
            Dataframe to suffix subject columns for, containing the results
            of students that belong to the same cohort year.
        year:
            The cohort year of the students of which the given results in the
            DataFrame belong to.
    Returns:
        The given DataFrame with the subject column suffixed with level.
    """
    # Calculate Secondary Level
    # match 'YYYY Result(s)' columns
    year_re = re.compile(r"(?P<year_result>\d{4}) Results?")
    year_matches = [year_re.match(c) for c in df.columns]
    # extract the year the subject was taken from the column name & column position
    year_results = [
        int(match.group("year_result")) for match in year_matches if match is not None
    ]
    # calculate secondary level of the student when the subject was taken
    grad_level = 4
    level_map = {
        # since year of cohort (year) is also the student's graduation year,
        # we can calculate the no. years till graduation by subtracting (year - year_result).
        # using this offset we can determine the secondary level of study the year
        # the result was recorded.
        year_result: f"S{grad_level - (year - year_result)}"
        for year_result in year_results
    }

    # Determine the subjects taken by secondary level
    # extract position of YYYY results columns
    year_col_idxs = [i for i, m in enumerate(year_matches) if m is not None]
    year_positions = list(zip(year_results, year_col_idxs))
    # add end bound for column positions
    year_positions.append((-1, len(df.columns)))
    # calculate start & end column positions of columns by year of results
    year_bounds = [
        (year_result, (begin + 1, end))
        for (year_result, begin), (_, end) in zip(
            year_positions[:-1], year_positions[1:]
        )
    ]
    # extract subject columns by year of results
    year_columns = {
        year_result: df.columns[begin:end] for year_result, (begin, end) in year_bounds
    }

    # Suffix subject column with level
    # strip .<DIGIT> suffix on subject column names English.3 -> English
    digit_re = re.compile(r"(?P<subject>.+)\.\d+$")

    def strip_digit(column: str) -> str:
        match = digit_re.match(column)
        if match is None:
            return column
        else:
            return match.group("subject")

    rename_map = {
        col: f"{strip_digit(col)} [{level_map[year_result]}]"
        for year_result in year_results
        for col in year_columns[year_result]
    }
    return df.rename(columns=rename_map)
