import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import (
    ALL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, EXTRA_FEATURES,
    NUMERIC_FEATURES, RANDOM_STATE, RAW_DATA, TARGET, TEST_SIZE,
)


def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


def split_data(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor(use_extra_features: bool = False) -> ColumnTransformer:
    """yes/no -> 0/1, furnishingstatus -> one-hot, numeric columns pass through.
    Tree models don't need scaling, so we skip it."""
    numeric = NUMERIC_FEATURES + (EXTRA_FEATURES if use_extra_features else [])
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("bin", OrdinalEncoder(categories=[["no", "yes"]] * len(BINARY_FEATURES)),
             BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
