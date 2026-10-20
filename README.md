# House Price Prediction

Predicting house prices from `Housing.csv` (545 houses, 12 features) with **Random Forest** and **XGBoost**, served through a **Flask** API with a simple web form.

## Project structure

```
data/raw/Housing.csv         original dataset
notebooks/eda.ipynb       exploratory data analysis
notebooks/model_evaluation.ipynb   metrics, residuals, feature importance
src/config.py                paths, seed, feature lists, modelling switches
src/data_preprocessing.py    loading, train/test split, encoding
src/features.py              optional engineered features
src/pipeline.py              builds the full sklearn pipeline
src/experiments.py           CV comparison of modelling variants
src/tune.py                  hyperparameter search -> models/best_params.json
src/train.py                 trains both models, saves them to models/
src/evaluate.py              MAE, RMSE, R2
src/predict.py               loads a saved model and predicts
app/main.py                  Flask app (/, /predict, /health)
app/schemas.py               input validation
app/templates/index.html     web form
tests/test_predict.py        tests
```

## How to run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.experiments         # (optional) compare modelling variants
python -m src.tune                # (optional) tune hyperparameters
python -m src.train               # train and save both models
python -m app.main                # start the app: http://127.0.0.1:5000
pytest                            # run tests (needs trained models for predict tests)
```

Run all commands from the project root and use `python -m ...` so the `src` imports work.

## API

`POST /predict` with JSON (field `model` is optional: `xgboost` by default, or `random_forest`):

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "parking": 2,
  "mainroad": "yes", "guestroom": "no", "basement": "no",
  "hotwaterheating": "no", "airconditioning": "yes", "prefarea": "yes",
  "furnishingstatus": "furnished", "model": "xgboost"
}'
```

Response: `{"model": "xgboost", "predicted_price": 8123456}`. Invalid input returns `400` with a list of errors; if the models have not been trained yet, `503`.

## Results

Test split (80/20, 109 houses), tuned models:

| Model         | MAE       | RMSE      | R2    |
|---------------|-----------|-----------|-------|
| Random Forest | 1,030,952 | 1,409,622 | 0.607 |
| XGBoost       |   965,819 | 1,333,776 | 0.648 |

Cross-validation on the train split during tuning gave R2 = 0.643 (Random Forest) and 0.646 (XGBoost), so the two models perform similarly.

Findings:
- Engineered features (`total_rooms`, `area_per_room`) and a log-transformed target did not improve either model (differences were within CV noise), so they are switched off in `src/config.py`.
- Hyperparameter tuning gave a small improvement (about +0.01 R2 in CV).
- R2 around 0.65 appears to be the limit for this small dataset.
