from src.config import ALL_FEATURES, BINARY_FEATURES

FURNISHING = {"furnished", "semi-furnished", "unfurnished"}
INT_RANGES = {
    "area": (100, 100000), "bedrooms": (1, 10), "bathrooms": (1, 10),
    "stories": (1, 10), "parking": (0, 10),
}


def validate(data: dict) -> list[str]:
    """Return a list of error messages (empty list = valid)."""
    errors = []
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        return [f"Missing fields: {', '.join(missing)}"]

    for f, (lo, hi) in INT_RANGES.items():
        v = data[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            errors.append(f"{f} must be a number between {lo} and {hi}")
    for f in BINARY_FEATURES:
        if data[f] not in ("yes", "no"):
            errors.append(f"{f} must be 'yes' or 'no'")
    if data["furnishingstatus"] not in FURNISHING:
        errors.append(f"furnishingstatus must be one of {sorted(FURNISHING)}")
    return errors
