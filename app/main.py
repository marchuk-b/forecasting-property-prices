from flask import Flask, jsonify, render_template, request

from app.schemas import validate
from src.predict import predict_price
from src.report import load_report, prediction_insights

app = Flask(__name__)
MODELS = ("random_forest", "xgboost")


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return render_template("index.html", page="predict")


@app.get("/models")
def models_page():
    return render_template("models.html", page="models")


@app.get("/dataset")
def dataset_page():
    return render_template("dataset.html", page="dataset")


# ---------------------------------------------------------------- API
@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/api/report")
def api_report():
    report = load_report()
    if report is None:
        return jsonify(error="Report not generated yet. Run: python -m src.train"), 503
    return jsonify(report)


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

    try:
        price = predict_price(data, model_name)
    except FileNotFoundError:
        return jsonify(error="Models are not trained yet. Run: python -m src.train"), 503

    response = {"model": model_name, "predicted_price": round(price)}

    insights = prediction_insights(price, model_name)
    if insights:
        response.update(insights)

    # what every available model says about this house
    comparison = {}
    for name in MODELS:
        try:
            comparison[name] = round(predict_price(data, name))
        except FileNotFoundError:
            pass
    if len(comparison) == 2:
        a, b = comparison.values()
        response["comparison"] = comparison
        response["model_difference_pct"] = round(abs(a - b) / ((a + b) / 2) * 100, 1)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
