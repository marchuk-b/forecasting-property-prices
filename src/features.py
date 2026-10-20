import pandas as pd

from src.config import EXTRA_FEATURES  # noqa: F401  (documented names)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Uses only input features (never the target)."""
    X = X.copy()
    X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
    X["area_per_room"] = X["area"] / X["bedrooms"]
    return X
