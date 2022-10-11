#
# Sort'in Hat
# Models
# Feature Extraction
#

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.engine.cursor import _NoResultMetaData

# feature extraction mappings
PSLE_SUBJECTS = ["EL", "MT", "Maths", "Sci", "HMT"]
PSLE_MAPPING = {
    "A*": 1,
    "A": 2,
    "B": 3,
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
}

HOUSING_MAPPING = {
    "Detached House": 1,
    "Semi-Detached House": 2,
    "Terrace": 3,
    "Private Flat/Apartment": 4,
    "Govt/Quasi-Govt Executive Flat": 5,
    "HDB/SAF/PSA/PUB 5_Room Flat": 6,
    "HDB/SAF/PSA/PUB 4_Room Flat": 7,
    "HDB/SAF/PSA/PUB 3_Room Flat": 8,
    "other": 9,
}

COURSE_MAPPING = {
    "Express": 1,
    "Normal Academic": 2,
    "Normal Technical": 3,
}

GENDER_MAPPING = {
    "Male": 1,
    "Female": 0,
}

CARDING_LEVELS = ["L3", "Y", "L4P", "L4", "YT", "TL3", "E3", "B4", "ET3", "Y+"]
CARDING_MAPPING = {level: True for level in CARDING_LEVELS}

SPORTS_LEVEL_MAPPING = {
    "1*": 1,
    "1A": 2,
    "1": 3,
    "2*": 4,
    "2A": 5,
    "2": 6,
    "3*": 7,
    "3A": 8,
    "3": 9,
}


def map_values(
    df: Union[pd.DataFrame, pd.Series],
    mapping: Dict[Any, Any],
    default: Any = pd.NA,
) -> Union[pd.DataFrame, pd.Series]:
    """Map values in the given DataFrame or Series using the given dictionary mapping."""
    replacer = lambda value: mapping[value] if value in mapping.keys() else default
    if isinstance(df, pd.DataFrame):
        return df.applymap(replacer)
    # otherwise, we are dealing with a series, which has an .map() instead of .applymap()
    return df.map(replacer)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature vector suitable for ML models from the given dataframe."""
    # extract categorical features using feature extraction mappings
    df["Sec4_CardingLevel"] = map_values(
        df["Sec4_CardingLevel"], CARDING_MAPPING, default=False
    )
    df["Gender"] = map_values(df["Gender"], GENDER_MAPPING)
    # rank missing sports levels in the last ranking (10)
    df["Sec4_SportsLevel"] = map_values(
        df["Sec4_SportsLevel"], SPORTS_LEVEL_MAPPING, default=10
    )
    df["Course"] = map_values(df["Course"], COURSE_MAPPING)
    df["ResidentialType"] = map_values(df["ResidentialType"], HOUSING_MAPPING)
    df[PSLE_SUBJECTS] = map_values(df[PSLE_SUBJECTS], PSLE_MAPPING)

    return df


def vectorize_features(df: pd.DataFrame) -> NDArray[np.float_]:
    """Vectorize dataframe into feature vectors."""
    return ColumnTransformer(
        transformers=[
            # one hot encode categorical columns
            (
                "categorical",
                OneHotEncoder(),
                df.select_dtypes(include="object").columns,
            ),
            # standard scale numeric columns
            ("numeric", StandardScaler(), df.select_dtypes(include="number").columns),
        ],
        remainder="passthrough",
    ).fit_transform(df)
