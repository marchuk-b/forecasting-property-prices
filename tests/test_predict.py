import pytest

from app.main import app
from src.config import MODELS_DIR

SAMPLE = {
    "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "parking": 2,
    "mainroad": "yes", "guestroom": "no", "basement": "no",
    "hotwaterheating": "no", "airconditioning": "yes", "prefarea": "yes",
    "furnishingstatus": "furnished",
}


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/health").json == {"status": "ok"}


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"House Price Prediction" in r.data


def test_not_json(client):
    assert client.post("/predict", data="hello").status_code == 400


def test_missing_fields(client):
    assert client.post("/predict", json={"area": 5000}).status_code == 400


def test_invalid_values(client):
    bad = {**SAMPLE, "furnishingstatus": "weird", "mainroad": "maybe", "bedrooms": 99}
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert len(r.json["errors"]) == 3


def test_unknown_model(client):
    assert client.post("/predict", json={**SAMPLE, "model": "linear"}).status_code == 400


@pytest.mark.parametrize("model", ["random_forest", "xgboost"])
def test_predict(client, model):
    if not (MODELS_DIR / f"{model}.joblib").exists():
        pytest.skip("model not trained - run: python -m src.train")
    r = client.post("/predict", json={**SAMPLE, "model": model})
    assert r.status_code == 200
    assert r.json["predicted_price"] > 0
