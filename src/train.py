"""Train Random Forest + XGBoost on the train split, evaluate on the test split,
save pipelines to models/. If models/best_params.json exists (from src.tune),
its hyperparameters are used instead of the defaults below.
"""
import json

import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import (
    BEST_PARAMS_FILE, LOG_TARGET, MODELS_DIR, RANDOM_STATE, USE_EXTRA_FEATURES,
)
from src.data_preprocessing import load_data, split_data
from src.evaluate import evaluate_model
from src.pipeline import build_pipeline


def get_models() -> dict:
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if BEST_PARAMS_FILE.exists():
        tuned = json.loads(BEST_PARAMS_FILE.read_text())
        for name, params in tuned.items():
            if name in models:
                models[name].set_params(**params)
        print(f"Using tuned hyperparameters from {BEST_PARAMS_FILE.name}\n")
    return models


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    MODELS_DIR.mkdir(exist_ok=True)

    results = {}
    for name, model in get_models().items():
        pipe = build_pipeline(model, USE_EXTRA_FEATURES, LOG_TARGET)
        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
        print(f"{name:14s}", {k: round(v, 3) for k, v in results[name].items()})

    best = max(results, key=lambda n: results[n]["r2"])
    print(f"\nBest model by R2: {best}")


if __name__ == "__main__":
    main()
