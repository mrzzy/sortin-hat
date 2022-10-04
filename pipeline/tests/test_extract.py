#
# Sort'in Hat
# Models
# Feature Extraction
#

from itertools import cycle, groupby, islice
from typing import Any, Dict

import numpy as np
import pandas as pd

from extract import (
    CARDING_LEVELS,
    COURSE_MAPPING,
    GENDER_MAPPING,
    HOUSING_MAPPING,
    PSLE_MAPPING,
    PSLE_SUBJECTS,
    SPORTS_LEVEL_MAPPING,
    extract_features,
    map_values,
)


def test_map_values():
    test_df = pd.DataFrame(
        {
            "A": np.arange(3),
            "B": np.arange(1, 4),
        }
    )
    mapping = {
        1.0: 1.0,
        3.0: 3.0,
        2.0: 99.0,
    }
    df = map_values(
        df=test_df,
        mapping=mapping,
    )

    # check: values mapped in all columns
    assert (df["A"].values[1:] == np.array([1.0, 99.0])).all()
    assert (df["B"].values[:3] == np.array([1.0, 99.0, 3.0])).all()

    # check: default applied when mapping is not defined
    assert pd.isna(df["A"][:1]).all()

    # check: map series
    assert (map_values(test_df["A"], mapping)[1:] == np.array([1.0, 99.0])).all()


def test_extract_features(dummy_data: Dict[str, Any]):
    n_keys = lambda mapping, n: list(islice(cycle(mapping.keys()), n))
    test_data = {
        # UNKNOWN added to verify mapped default values
        "Sec4_CardingLevel": CARDING_LEVELS[:2] + ["UNKNOWN"],
        "Gender": n_keys(GENDER_MAPPING, 3),
        "Sec4_SportsLevel": n_keys(SPORTS_LEVEL_MAPPING, 2) + ["UNKNOWN"],
        "Course": n_keys(COURSE_MAPPING, 3),
        "ResidentialType": n_keys(HOUSING_MAPPING, 3),
    }
    test_data.update({subject: n_keys(PSLE_MAPPING, 3) for subject in PSLE_SUBJECTS})
    test_data.update(dummy_data)

    df = extract_features(pd.DataFrame(test_data))

    # check: carding level extraction
    assert (df["Sec4_CardingLevel"] == np.array([True, True, False])).all()
    # check: gender extraction
    assert (df["Gender"] == np.array([1, 0, 1])).all()
    # check: sports level extraction
    assert (df["Sec4_SportsLevel"].values[:2] == np.array([1, 2])).all() and df[
        "Sec4_SportsLevel"
    ][2] == 10
    # check: course extraction
    assert (df["Course"] == np.array([1, 2, 3])).all()
    # check: residential extraction
    assert (df["ResidentialType"] == np.array([1, 2, 3])).all()
    # check: PSLE subjects extraction
    assert (
        (
            df[PSLE_SUBJECTS]
            == pd.DataFrame({subject: np.array([1, 2, 3]) for subject in PSLE_SUBJECTS})
        )
        .all()
        .all()
    )

    # check: one hot encoding of categorical columns
    cat_cols = [name for name in df.columns if "dummy_cat" in name]
    assert len(cat_cols) > 0
    for key, cols in groupby(cat_cols, lambda name: name[:-2]):
        assert all([f"{key}_{i}" == name for i, name in enumerate(cols)])
