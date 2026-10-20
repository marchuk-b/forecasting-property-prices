"""Compare modelling variants with repeated 5-fold CV on the whole dataset.
Run: python -m src.experiments
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate

from src.config import ALL_FEATURES, RANDOM_STATE, TARGET
from src.data_preprocessing import load_data
from src.pipeline import build_pipeline


def get_models():
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
    except ImportError:
        print("xgboost not installed - skipping it")
    return models


VARIANTS = {
    "baseline":        dict(use_extra_features=False, log_target=False),
    "+features":       dict(use_extra_features=True,  log_target=False),
    "+log_target":     dict(use_extra_features=False, log_target=True),
    "+features+log":   dict(use_extra_features=True,  log_target=True),
}


def main():
    df = load_data()
    X, y = df[ALL_FEATURES], df[TARGET]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    rows = []
    for mname, model in get_models().items():
        for vname, flags in VARIANTS.items():
            res = cross_validate(build_pipeline(model, **flags), X, y, cv=cv, scoring=scoring)
            rows.append({
                "model": mname, "variant": vname,
                "r2": res["test_r2"].mean(), "r2_std": res["test_r2"].std(),
                "mae": -res["test_mae"].mean(),
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:,.3f}".format)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
