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

from clean import clean_extract, clean_p6
from extract import (
    encode_psle,
    encode_sports_level,
    get_carding,
    get_course_tier,
    get_gender,
    get_housing,
)
from transform import suffix_subject_level


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
    df = clean_extract(df)
    # suffix subject columns with secondary level
    df = suffix_subject_level(df, year)
    # drop YYYY result columns which is empty
    df = df.drop(columns=[col for col in df.columns if "Result" in col])
    # reduce carding level to boolean flag
    # Extract Features
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
