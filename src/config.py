from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "Housing.csv"
MODELS_DIR = ROOT / "models"

TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_FEATURES = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]
CATEGORICAL_FEATURES = ["furnishingstatus"]  # furnished / semi-furnished / unfurnished

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

EXTRA_FEATURES = ["total_rooms", "area_per_room"]  # created in src/features.py

# Modelling switches (decide using src/experiments.py)
USE_EXTRA_FEATURES = False
LOG_TARGET = False

BEST_PARAMS_FILE = MODELS_DIR / "best_params.json"
