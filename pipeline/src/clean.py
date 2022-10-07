#
# Sort'in Hat
# Models
# Prepare Data
#

from typing import Any, Dict

import numpy as np
import pandas as pd

from extract import PSLE_SUBJECTS

P6_COLUMNS = [
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


def clean_p6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given 'P6 Screening Master Template' data.
    Warning: Drops rows that do not fulfill data quality requirements.

    Args:
        df:
            Dataframe to clean.
    Returns:
        The prepared dataframe for model training.
    """
    # Clean dataframe
    # fix name of, drop missing values & fix type serial no. column
    df = df.rename(columns={"Unnamed: 0": "Serial number"})
    df["Serial number"] = df["Serial number"].replace("x", np.nan)
    # TODO(mrzzy): add warning
    df = df.dropna(subset=["Serial number"])
    df["Serial number"] = df["Serial number"].astype(np.int_)

    # drop unknown strings in Q1 M
    df["Q1 M"] = pd.to_numeric(df["Q1 M"], errors="coerce")
    # drop rows with no or Q1 M
    # TODO(mrzzy): add warning
    df = df.dropna(subset=["Q1 M"])

    # Select only required columns
    df = df[P6_COLUMNS]

    return df


def clean_extract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given 'Template for Data Extraction' data
    Warning: Drops rows that do not fulfill data quality requirements.

    Args:
        df:
            Dataframe to cleaned.
    Returns:
        The cleaned dataframe for model training.
    """
    # Clean dataframe
    # parse "-" / 0 for missing value
    df = df.replace("-", np.nan).replace("0", np.nan).replace(0, np.nan)
    # override types explicitly where pandas type detection fails
    df["Sec4_SportsLevel"] = df["Sec4_SportsLevel"].fillna("").astype(np.str_)
    df[PSLE_SUBJECTS] = df[PSLE_SUBJECTS].astype(np.str_)
    df["Serial number"] = pd.to_numeric(df["Serial number"], errors="coerce")

    # drop rows with missing serial no.
    # TODO(mrzzy): add warning
    df = df.dropna(subset=["Serial number"])
    # drop rows with non-string course
    df = df[[isinstance(v, str) for v in df["Course"]]]

    # strip leading 'L' from level as some level's are missing leading 'L'
    df["Sec4_SportsLevel"] = df["Sec4_SportsLevel"].str.replace("L", "")

    # rename 'Sec4_BoardingStatus' to 'BoardingStatus' as they appear to refer
    # to the same thing
    if "Sec4_BoardingStatus" in df.columns:
        df = df.rename(columns={"Sec4_BoardingStatus": "BoardingStatus"})

    return df
