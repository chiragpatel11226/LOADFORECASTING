"""
app.py
------
Flask backend for the Power Demand Forecast dashboard.

Responsibilities:
1. Serve the single-page dashboard (templates/index.html).
2. Serve pre-computed chart data (built once by src/train_model.py) so the
   dashboard loads instantly without recomputing aggregates on every request.
3. Expose a /api/predict endpoint that turns a user-chosen date + hour into
   the same feature vector used at training time, then runs it through the
   trained Random Forest model.

Run with:  python app.py   (see README.md for full setup instructions)
"""

import json
import sys
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, "src")
from preprocess import FEATURE_COLUMNS  # noqa: E402  (import after sys.path tweak)

app = Flask(__name__)

MODEL = joblib.load("models/demand_model.pkl")

with open("models/metrics.json") as f:
    METRICS = json.load(f)

with open("models/feature_importance.json") as f:
    FEATURE_IMPORTANCE = json.load(f)

with open("models/dashboard_data.json") as f:
    DASHBOARD_DATA = json.load(f)

DATASET_START = datetime.strptime(DASHBOARD_DATA["summary"]["dataset_start"], "%Y-%m-%d")


def build_feature_vector(target_date: date, hour: int) -> list:
    """Build a feature row for one (date, hour) pair, using EXACTLY the same
    formulas as src/preprocess.py's add_calendar_features(). Keeping the
    formulas identical between training and inference is the single most
    important correctness rule for a deployed ML model -- a mismatch here
    ("training skew") is a very common real-world bug.
    """
    dt = datetime(target_date.year, target_date.month, target_date.day, hour)
    day_of_year = dt.timetuple().tm_yday
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    trend_days = (dt - DATASET_START).total_seconds() / 86400

    row = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        "dow_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dayofweek / 7),
        "is_weekend": is_weekend,
        "day_of_year": day_of_year,
        "trend_days": trend_days,
    }
    # Returned as a single-row DataFrame (not a bare list) so scikit-learn
    # sees the same column names it was trained with -- avoids a
    # "X does not have valid feature names" warning and is safer if the
    # model is ever swapped for one that validates input schemas.
    return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """All the pre-aggregated series the charts need, in one payload."""
    return jsonify(DASHBOARD_DATA)


@app.route("/api/metrics")
def metrics():
    """Model comparison table + error metrics."""
    return jsonify(METRICS)


@app.route("/api/feature-importance")
def feature_importance():
    return jsonify(FEATURE_IMPORTANCE)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        target_date = datetime.strptime(payload["date"], "%Y-%m-%d").date()
        hour = int(payload["hour"])
        if not (0 <= hour <= 23):
            raise ValueError("hour must be between 0 and 23")
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    features = build_feature_vector(target_date, hour)
    prediction = float(MODEL.predict(features)[0])

    dayofweek_name = datetime(target_date.year, target_date.month, target_date.day).strftime("%A")

    return jsonify({
        "predicted_mw": round(prediction, 1),
        "date": payload["date"],
        "hour": hour,
        "day_of_week": dayofweek_name,
        "is_weekend": dayofweek_name in ("Saturday", "Sunday"),
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
