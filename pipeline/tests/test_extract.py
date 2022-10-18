#
# Sort'in Hat
# Models
# Feature Extraction
#

from itertools import cycle, islice
from typing import Any, Dict
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from extract import (
    PSLE_SUBJECTS,
    extract_features,
    featurize_dataset,
    map_values,
    vectorize_features,
)


@pytest.mark.unit
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
    assert (df["A"].values[1:] == np.array([1.0, 99.0])).all()  # type: ignore
    assert (df["B"].values[:3] == np.array([1.0, 99.0, 3.0])).all()  # type: ignore

    # check: default applied when mapping is not defined
    assert pd.isna(df["A"][:1]).all()

    # check: map series
    assert (map_values(test_df["A"], mapping)[1:] == np.array([1.0, 99.0])).all()


@pytest.mark.unit
def test_extract_features(extract_df: pd.DataFrame):
    df = extract_features(extract_df)

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


@pytest.mark.unit
def test_vectorize_features(dummy_data: Dict[str, Any]):
    df = pd.DataFrame(dummy_data)

    # check: one hot encoding applied to 2x categorical columns
    features = vectorize_features(df.select_dtypes(include="object"))
    assert features.shape[-1] == 2 * 3
    assert ((features == 1) | (features == 0)).all()
    # check: standard deviation scaling applied to numeric columns
    numeric_df = df.select_dtypes(include="number")
    features = vectorize_features(numeric_df)
    assert (features == StandardScaler().fit_transform(numeric_df.values)).all()


@pytest.mark.unit
def test_featurize_dataset(dummy_data: Dict[str, Any]):
    # use first column of dummy datafram as target
    df = pd.DataFrame(dummy_data)
    target, feature_df = df.columns.to_list()[0], df.iloc[:, 1:]

    with mock.patch(
        "extract.extract_features", return_value=feature_df
    ) as extract_features:
        featurize_dataset(df, target)
        extract_features.assert_called_once()
        assert (extract_features.call_args[0][0] == feature_df).all().all()
