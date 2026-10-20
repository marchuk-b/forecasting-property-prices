import pytest

from app.main import app

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


def test_missing_fields(client):
    assert client.post("/predict", json={"area": 5000}).status_code == 400


def test_predict(client):  # requires trained models (run python -m src.train first)
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json["predicted_price"] > 0
