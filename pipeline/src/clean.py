#
# Sort'in Hat
# Models
# Prepare Data
#

import numpy as np
import pandas as pd

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
    Clean the given 'P6 Screening' data

    Args:
        df:
            Dataframe to clean.
    Returns:
        The prepared dataframe for model training.
    """
    # Clean dataframe
    # fix name of serial no. column
    df = df.rename(columns={"Unnamed: 0": "Serial number"})

    # select required columns
    df = df[P6_COLUMNS]

    # fix question Q1 M data type by converting unknown strings to NaN
    # TODO(mrzzy): rectify data upstream "x+C75:C80" string in data
    df["Q1 M"] = pd.to_numeric(df["Q1 M"], errors="coerce")

    ## drop students with missing serial nos
    df["Serial number"] = df["Serial number"].replace("x", np.nan)
    df = df[~df["Serial number"].isna()]
    df["Serial number"] = df["Serial number"].astype(np.int_)

    return df


def clean_extract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given 'Data Extraction' dataa.

    Args:
        df:
            Dataframe to cleaned.

    Returns:
        The cleaned dataframe for model training.
    """
    # Clean dataframe
    # parse "-" / 0 for missing value
    df = df.replace("-", np.nan).replace("0", np.nan).replace(0, np.nan)
    # enforce integer type for serial numbers
    df["Serial number"] = df["Serial number"].astype(np.int_)
    # rename 'Sec4_BoardingStatus' to 'BoardingStatus' as they appear to refer
    # to the same thing
    if "Sec4_BoardingStatus" in df.columns:
        df = df.rename(columns={"Sec4_BoardingStatus": "BoardingStatus"})

    return df
