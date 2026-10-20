import joblib
import pandas as pd

from src.config import ALL_FEATURES, MODELS_DIR

_cache = {}


def load_model(name: str = "xgboost"):
    if name not in _cache:
        _cache[name] = joblib.load(MODELS_DIR / f"{name}.joblib")
    return _cache[name]


def predict_price(features: dict, model_name: str = "xgboost") -> float:
    df = pd.DataFrame([features])[ALL_FEATURES]
    return float(load_model(model_name).predict(df)[0])
