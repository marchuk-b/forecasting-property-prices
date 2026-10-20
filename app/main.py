from flask import Flask, jsonify, request

from app.schemas import validate
from src.predict import predict_price

app = Flask(__name__)
MODELS = ("random_forest", "xgboost")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object"), 400

    model_name = data.pop("model", "xgboost")
    if model_name not in MODELS:
        return jsonify(error=f"model must be one of {MODELS}"), 400

    errors = validate(data)
    if errors:
        return jsonify(errors=errors), 400

    price = predict_price(data, model_name)
    return jsonify(model=model_name, predicted_price=round(price))


if __name__ == "__main__":
    app.run(debug=True)
