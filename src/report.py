"""Builds models/report.json: all statistics shown on the web pages
(dataset overview, model evaluation, prediction intervals).

Run:  python -m src.report        (also called automatically at the end of src.train)
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate

from src.config import (
    ALL_FEATURES, BEST_PARAMS_FILE, BINARY_FEATURES, LOG_TARGET, MODELS_DIR,
    NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, USE_EXTRA_FEATURES,
)
from src.data_preprocessing import load_data, split_data
from src.evaluate import evaluate_model
from src.pipeline import build_pipeline

REPORT_FILE = MODELS_DIR / "report.json"
MODEL_NAMES = ("random_forest", "xgboost")
INTERVAL_LEVELS = (80, 90)

COLUMN_INFO = {
    "price": "House price (the value we predict)",
    "area": "Total area of the house (sq ft)",
    "bedrooms": "Number of bedrooms",
    "bathrooms": "Number of bathrooms",
    "stories": "Number of floors (stories)",
    "mainroad": "Connected to a main road",
    "guestroom": "Has a guest room",
    "basement": "Has a basement",
    "hotwaterheating": "Has a hot water heating system",
    "airconditioning": "Has air conditioning",
    "parking": "Number of car parking spaces",
    "prefarea": "Located in a preferred area",
    "furnishingstatus": "Furnished, semi-furnished or unfurnished",
}


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialize {type(o)}")


def _num_summary(s: pd.Series) -> dict:
    return {
        "min": s.min(), "q25": s.quantile(0.25), "median": s.median(),
        "mean": s.mean(), "q75": s.quantile(0.75), "max": s.max(), "std": s.std(),
    }


# --------------------------------------------------------------------------- dataset
def dataset_info(df: pd.DataFrame) -> dict:
    price = df[TARGET]
    counts, edges = np.histogram(price, bins=20)

    columns = []
    for col in [TARGET] + ALL_FEATURES:
        if col == TARGET:
            kind = "target (number)"
        elif col in NUMERIC_FEATURES:
            kind = "number"
        elif col in BINARY_FEATURES:
            kind = "yes / no"
        else:
            kind = "category"
        item = {"name": col, "kind": kind, "description": COLUMN_INFO[col]}
        if pd.api.types.is_numeric_dtype(df[col]):
            item["summary"] = _num_summary(df[col])
        else:
            g = df.groupby(col)[TARGET].agg(["count", "median"])
            item["values"] = [
                {"value": v, "count": int(r["count"]), "median_price": float(r["median"])}
                for v, r in g.iterrows()
            ]
        columns.append(item)

    corr = df[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    return {
        "rows": len(df), "n_features": len(ALL_FEATURES),
        "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
        "price": _num_summary(price),
        "price_hist": {"edges": edges, "counts": counts},
        "columns": columns,
        "correlations": corr.to_dict(),
    }


# --------------------------------------------------------------------------- models
def models_info(df: pd.DataFrame) -> dict:
    from src.train import get_models  # imported here to avoid a circular import

    X_train, X_test, y_train, y_test = split_data(df)
    X_all, y_all = df[ALL_FEATURES], df[TARGET]
    tuned_params = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    cv_repeated = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cv_oof = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    out = {}
    for name, model in get_models().items():
        fitted = joblib.load(MODELS_DIR / f"{name}.joblib")

        # ---- test split
        pred = fitted.predict(X_test)
        test = evaluate_model(fitted, X_test, y_test)
        test["mape"] = float(np.mean(np.abs(y_test - pred) / y_test) * 100)
        resid_counts, resid_edges = np.histogram(y_test - pred, bins=15)

        # ---- cross-validation on all data
        pipe = build_pipeline(model, USE_EXTRA_FEATURES, LOG_TARGET)
        cv = cross_validate(pipe, X_all, y_all, cv=cv_repeated, scoring=scoring)

        # ---- prediction intervals from out-of-fold relative errors
        oof = cross_val_predict(pipe, X_all, y_all, cv=cv_oof)
        rel = (y_all.to_numpy() - oof) / oof            # error as a share of the prediction
        interval = {}
        for level in INTERVAL_LEVELS:
            tail = (1 - level / 100) / 2
            lo, hi = np.quantile(rel, [tail, 1 - tail])
            interval[str(level)] = [float(lo), float(hi)]

        # ---- permutation importance (test split, original columns)
        pi = permutation_importance(fitted, X_test, y_test, n_repeats=20,
                                    random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
        importance = sorted(
            ({"feature": f, "mean": float(m), "std": float(s)}
             for f, m, s in zip(X_test.columns, pi.importances_mean, pi.importances_std)),
            key=lambda d: d["mean"], reverse=True,
        )

        shown = tuned_params.get(name) or {
            k: v for k, v in model.get_params().items()
            if k in ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
                     "subsample", "colsample_bytree")
        }
        out[name] = {
            "test": test,
            "cv": {
                "r2_mean": cv["test_r2"].mean(), "r2_std": cv["test_r2"].std(),
                "mae_mean": -cv["test_mae"].mean(), "rmse_mean": -cv["test_rmse"].mean(),
            },
            "interval": interval,
            "scatter": [[float(a), float(p)] for a, p in zip(y_test, pred)],
            "residual_hist": {"edges": resid_edges, "counts": resid_counts},
            "importance": importance,
            "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in shown.items()},
            "tuned": name in tuned_params,
        }
    return out


# --------------------------------------------------------------------------- build / load
def build_report() -> dict:
    df = load_data()
    X_train, X_test, _, _ = split_data(df)
    report = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "train_rows": len(X_train), "test_rows": len(X_test), "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "use_extra_features": USE_EXTRA_FEATURES, "log_target": LOG_TARGET,
        },
        "dataset": dataset_info(df),
        "models": models_info(df),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, default=_default), encoding="utf-8")
    print(f"Report saved to {REPORT_FILE}")
    return report


_report_cache = {"mtime": None, "data": None}


def load_report():
    """Return the saved report (re-read automatically if the file changed) or None."""
    if not REPORT_FILE.exists():
        return None
    mtime = REPORT_FILE.stat().st_mtime
    if _report_cache["mtime"] != mtime:
        _report_cache.update(mtime=mtime, data=json.loads(REPORT_FILE.read_text(encoding="utf-8")))
    return _report_cache["data"]


_prices_cache = {}


def _prices() -> np.ndarray:
    if "p" not in _prices_cache:
        _prices_cache["p"] = np.sort(load_data()[TARGET].to_numpy())
    return _prices_cache["p"]


def prediction_insights(price: float, model_name: str):
    """Statistics that go with one prediction, or None if no report exists yet."""
    report = load_report()
    if report is None or model_name not in report["models"]:
        return None
    m = report["models"][model_name]
    prices = _prices()
    interval = {
        level: [max(0, round(price * (1 + lo))), round(price * (1 + hi))]
        for level, (lo, hi) in m["interval"].items()
    }
    return {
        "interval": interval,
        "quality": {
            "cv_r2": m["cv"]["r2_mean"], "cv_mae": m["cv"]["mae_mean"],
            "test_r2": m["test"]["r2"], "test_mae": m["test"]["mae"],
        },
        "percentile": round(float(np.searchsorted(prices, price, side="left") / len(prices) * 100), 1),
        "dataset_price": {
            "min": float(prices[0]), "median": float(np.median(prices)), "max": float(prices[-1]),
        },
    }


if __name__ == "__main__":
    build_report()
