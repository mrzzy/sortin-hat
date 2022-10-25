#
# Sort'in Hat
# Models
# Prepare Data
#

import numpy as np
import pandas as pd

from extract import PSLE_SUBJECTS

SERIAL_NO = "Serial number"

# P6 screening data specific columns: omits serial no. column
P6_COLUMNS = [
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
    Warning: Drops rows with missing Serial number.

    Args:
        df:
            Dataframe to clean.
    Returns:
        The prepared dataframe for model training.
    """
    # Clean dataframe
    # fix name of serial number, drop missing values
    df = df.rename(columns={"Unnamed: 0": SERIAL_NO})
    df[SERIAL_NO] = df[SERIAL_NO].replace("x", np.nan)
    # TODO(mrzzy): add warning
    df = df.dropna(subset=[SERIAL_NO])

    # coerce unknown strings in Q1 M to NaN
    df["Q1 M"] = pd.to_numeric(df["Q1 M"], errors="coerce")

    # Select only required columns
    df = df[[SERIAL_NO] + P6_COLUMNS]

    # enforce data schema by converting type to expected type
    df[P6_COLUMNS] = df[P6_COLUMNS].astype(np.float_)
    df[SERIAL_NO] = df[SERIAL_NO].astype(np.int_)

    return df


EXTRACT_SCHEMA = {
    SERIAL_NO: np.int_,
    "Gender": np.str_,
    "Race": np.str_,
    "Academy": np.str_,
    "Sec4_SportsLevel": np.str_,
    "SportsLevelYear": np.float_,
    "Sec4_CardingLevel": np.str_,
    "CardingLevelYear": np.float_,
    "MidStreamer": np.str_,
    "Class": np.str_,
    "Course": np.str_,
    "TScore": np.float_,
    "EducationalLevel_Father": np.str_,
    "EducationalLevel_Mother": np.str_,
    "ResidentialType": np.str_,
}


def clean_extract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given 'Template for Data Extraction' data
    Warning: Drops rows that do not fulfill data quality requirements)

    Args:
        df:
            Dataframe to cleaned.
    Returns:
        The cleaned dataframe for model training.
    """
    # Clean dataframe
    # parse "-" / 0 for missing value
    df = df.replace("-", np.nan).replace("0", np.nan).replace(0, np.nan)

    # parse & drop rows with missing serial no.
    # TODO(mrzzy): add warning
    df[SERIAL_NO] = pd.to_numeric(df[SERIAL_NO], errors="coerce")
    df = df.dropna(subset=[SERIAL_NO])

    # override types explicitly where pandas type detection fails
    df = df.astype(EXTRACT_SCHEMA)
    df[PSLE_SUBJECTS] = df[PSLE_SUBJECTS].astype(np.str_)

    # strip leading 'L' from level as some level's are missing leading 'L'
    df["Sec4_SportsLevel"] = df["Sec4_SportsLevel"].str.replace("L", "")

    # rename 'Sec4_BoardingStatus' to 'BoardingStatus' as they appear to refer
    # to the same thing
    if "Sec4_BoardingStatus" in df.columns:
        df = df.rename(columns={"Sec4_BoardingStatus": "BoardingStatus"})

    return df
