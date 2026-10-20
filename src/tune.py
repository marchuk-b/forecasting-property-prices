"""Hyperparameter search (5-fold CV on the TRAIN split only).
Run: python -m src.tune      ->  saves models/best_params.json
Then run: python -m src.train
"""
import json

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.config import (
    BEST_PARAMS_FILE, LOG_TARGET, MODELS_DIR, RANDOM_STATE, USE_EXTRA_FEATURES,
)
from src.data_preprocessing import load_data, split_data
from src.pipeline import build_pipeline, param_prefix

N_ITER = 40


def search_spaces():
    spaces = {
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": randint(100, 600),
                "max_depth": [None, 4, 6, 8, 12],
                "min_samples_leaf": randint(1, 8),
                "min_samples_split": randint(2, 12),
                "max_features": [1.0, 0.7, 0.5, "sqrt"],
            },
        )
    }
    try:
        from xgboost import XGBRegressor
        spaces["xgboost"] = (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": randint(100, 800),
                "learning_rate": uniform(0.01, 0.14),
                "max_depth": randint(2, 6),
                "min_child_weight": randint(1, 8),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": uniform(0.5, 5),
            },
        )
    except ImportError:
        print("xgboost not installed - skipping it")
    return spaces


def main(n_iter: int = N_ITER):
    df = load_data()
    X_train, _, y_train, _ = split_data(df)   # test set stays untouched
    prefix = param_prefix(LOG_TARGET)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best = {}
    for name, (model, dist) in search_spaces().items():
        pipe = build_pipeline(model, USE_EXTRA_FEATURES, LOG_TARGET)
        search = RandomizedSearchCV(
            pipe, {prefix + k: v for k, v in dist.items()},
            n_iter=n_iter, cv=cv, scoring="r2",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best[name] = {k[len(prefix):]: (v.item() if hasattr(v, "item") else v)
                      for k, v in search.best_params_.items()}
        print(f"{name}: CV R2 = {search.best_score_:.3f}\n  {best[name]}")

    MODELS_DIR.mkdir(exist_ok=True)
    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nSaved to {BEST_PARAMS_FILE}")


if __name__ == "__main__":
    main()
