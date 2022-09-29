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

from clean import clean_p6
from extract import (
    encode_psle,
    encode_sports_level,
    get_carding,
    get_course_tier,
    get_gender,
    get_housing,
)
from transform import suffix_subject_level


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
        pd.concat([clean_p6(df) for df in p6_dfs]).reset_index().drop(columns="index")
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
