#
# Sort'in Hat
# Models
# Prepare Data
#

from pathlib import Path
from typing import Iterable, List, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.base import re

from extract import (
    encode_psle,
    encode_sports_level,
    get_carding,
    get_gender,
    get_course_tier,
    get_housing,
)

# TODO(mrzzy): drop outliers? currently unused
def drop_outliers(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop outliers identified in the given dataframe from the given columns.

    Outliers are identified as data values that exceed the 1.5 * the IQR range
    from 25 percentile (Q1) or 75 percentile (Q3).

    Args:
        df:
            DataFrame to look for outliers in.
        cols:
            Check for outliers in the data values of this list of columns.
    Returns:
        DataFrame, with the rows identified as outliers, dropped.
    """
    # compute lower, upper quantile & inter-quantile range (IQR)
    lower_quantile, upper_quantile = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = upper_quantile - lower_quantile

    # filter out rows identified as outliers: consider all rows with at least
    # 1 data value outside the acceptable range for the range as outliers.
    allowed_deviation = 1.5 * iqr
    outliers = (
        (df[cols] < (lower_quantile - allowed_deviation))
        | (df[cols] > (upper_quantile + allowed_deviation))
    ).any(axis=1)

    return df[~outliers].copy()


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


def prepare_extract(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Prepare the given 'Data Extraction' dataframe for model training.

    Args:
        df:
            Dataframe to prepare.
        year:
            The year from which the data in the given dataframe is from.

    Returns:
        The prepared dataframe for model training.
    """
    # Clean dataframe
    # parse "-" / 0 for missing value
    df = df.replace("-", np.nan).replace("0", np.nan).replace(0, np.nan)
    # suffix subject columns with secondary level
    df = suffix_subject_level(df, year)
    # drop YYYY result columns which is empty
    df = df.drop(columns=[col for col in df.columns if "Result" in col])

    # reduce carding level to boolean flag
    df["Sec4_CardingLevel"] = (
        df["Sec4_CardingLevel"]
        .replace(
            {
                l: True
                for l in ["L3", "Y", "L4P", "L4", "YT", "TL3", "E3", "B4", "ET3", "Y+"]
            }
        )
        .replace(np.nan, False)
    )
    # enforce integer type for serial numbers
    df["Serial number"] = df["Serial number"].astype(np.int_)
    # rename 'Sec4_BoardingStatus' to 'BoardingStatus' as they appear to refer
    # to the same thing
    if "Sec4_BoardingStatus" in df.columns:
        df = df.rename(columns={"Sec4_BoardingStatus": "BoardingStatus"})

    # Extract Features
    # add year column to mark the year the data originated from
    df["year"] = year
    # extract categorical features using custom feature extraction functions
    extract_fns = {
        "Gender": get_gender,
        "Sec4_CardingLevel": get_carding,
        "Sec4_SportsLevel": encode_sports_level,
        "Course": get_course_tier,
        "ResidentialType": get_housing,
    }
    extract_fns.update(
        {subject: encode_psle for subject in ["EL", "MT", "Maths", "Sci", "HMT"]}
    )
    df[list(extract_fns.keys())] = df.transform(extract_fns)

    # replace rest of the categorical columns with a one-hot encoding as there
    # are no ordering dependencies between levels
    category_cols = df.dtypes[df.dtypes == np.dtype("O")].index
    encodings = pd.get_dummies(df[category_cols])
    df = df.drop(columns=category_cols).join(encodings)

    return df.sort_values(by="year")


def prepare_p6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the given 'P6 Screening' dataframe for model training.

    Args:
        df:
            Dataframe to prepare.

    Returns:
        The prepared dataframe for model training.
    """
    # Clean dataframe
    # fix name of serial no. column
    df = df.rename(columns={"Unnamed: 0": "Serial number"})

    # restrict to required columns only
    cols = [
        "Serial number",
        "Q1 M",
        "Q1F",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7",
        "Q8a",
        "Q8b",
        "Q8c",
        "Percentage (%)",
        "Percentage (%).1",
        "Percentage (%).2",
        "Percentage (%).3",
        "Percentage (%).4",
        "Percentage (%).5",
        "Q1.6",
        "Q2a",
        "Q2b",
        "Q2c",
        "Q2d",
        "Q2e",
        "Q2f",
        "Q2g",
        "Q2h",
        "Q2i",
        "Q2j",
        "Q2k",
        "Q3.7",
        "Q4a",
        "Q4b",
        "Q4c",
        "Q4d",
        "Q4e",
        "Q4f",
        "Q4g",
        "Q4h",
        "Q4i",
        "Q4j",
        "Q4k",
    ]
    df = df[cols]

    # fix question Q1 M data type by converting unknown strings to NaN
    # TODO(mrzzy): rectify data upstream "x+C75:C80" string in data
    df["Q1 M"] = pd.to_numeric(df["Q1 M"], errors="coerce")

    ## drop students with missing serial nos
    df["Serial number"] = df["Serial number"].replace("x", np.nan)
    df = df[~df["Serial number"].isna()]
    df["Serial number"] = df["Serial number"].astype(np.int_)

    return df


def load_dataset(
    data_dir: Path, extract_regex: re.Pattern, p6_regex: re.Pattern
) -> pd.DataFrame:
    """
    Load the Dataset from data files matching the given regex patterns.

    Args:
        data_dir:
            Path to the data directory to look for data files & load.
        extract_regex:
            Regex matching 'Data Extraction' data files to load.
        p6_regex:
            Regex matching 'P6 Screening' data files to load.
    Returns:
        Dataframe representing the loaded Dataset.
    """
    # extract absolute paths to matching data files
    paths = [p.resolve(strict=True) for p in data_dir.rglob("*")]
    extract_paths = [p for p in paths if extract_regex.match(p.name)]
    p6_paths = [p for p in paths if p6_regex.match(p.name)]

    # parse data age as year from paths
    extract_years = [
        int(cast(re.Match, extract_regex.match(p.name)).group("year"))
        for p in extract_paths
    ]

    # read data files as dataframes
    extract_dfs = [pd.read_excel(str(p)) for p in extract_paths]
    # use second row as a header for p6 screening data
    p6_dfs = [pd.read_excel(str(p), header=1) for p in p6_paths]

    # prepare dataframes for model training
    extract_df = (
        pd.concat(
            [prepare_extract(df, year) for df, year in zip(extract_dfs, extract_years)]
        )
        .reset_index()
        .drop(columns="index")
    )
    p6_df = (
        pd.concat([prepare_p6(df) for df in p6_dfs]).reset_index().drop(columns="index")
    )
    # join p6 screening data to the rest of the data extract
    df = pd.merge(extract_df, p6_df, how="left", on="Serial number")
    # serial no. column no longer needed post join.
    dataset_df = df.drop(columns=["Serial number"])

    return dataset_df


def segment_dataset(
    df: pd.DataFrame,
) -> Iterable[Tuple[int, str, pd.DataFrame, pd.Series]]:
    """Segments the given dataset into features & targets for training models to predict each subject & level.

    Segments the dataset to input features & output targets for training
    models to predict subjects by secondary school level (S1-S4).

    Args:
        df: Pandas dataframe of the dataset to segment.
    Returns:
        Generator producing (level, subject, features, labels) for each subject by level.
    """
    grad_level = 4

    # compile features for target prediction level (Secondary 1 -> 4)
    for level in range(1, grad_level + 1):
        future_levels = [f"[S{l}]" for l in range(level, grad_level + 1)]

        for subject in [col for col in df.columns if f"[S{level}]" in col]:
            # drop rows with NAN on target subject scores
            subject_df = df[~df[subject].isna()]

            # drop subjects taken in future levels to prevent time leakage
            # in input features
            features_df = subject_df[
                [
                    col
                    for col in df.columns
                    if not any([l in col for l in future_levels])
                ]
            ]

            # strip level suffix from subject name
            subject_name = cast(str, subject.replace(f"[S{level}]", "").rstrip())

            yield (level, subject_name, features_df, subject_df[subject])
