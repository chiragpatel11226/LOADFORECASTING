"""
train_model.py
---------------
Trains a demand-forecasting model on the calendar features built in
preprocess.py, evaluates it honestly, and exports everything the Flask app
needs to run without retraining on every request:

  models/demand_model.pkl       -> the fitted scikit-learn model
  models/metrics.json           -> error metrics + model comparison table
  models/feature_importance.json-> for the dashboard's feature-importance chart
  models/dashboard_data.json    -> pre-aggregated series for the charts

WHY THIS TRAIN/TEST SPLIT MATTERS (be ready to explain this in an interview):
A random 80/20 shuffle-split is the wrong choice for time-series data,
because it would let the model "peek" at hours right next to a test hour
(e.g. train on 2pm and 4pm, test on 3pm of the same day) -- that leaks
information and makes accuracy look better than it really is.

Instead we split by *time*: train on the full year 2017, and test on
Jan-Mar 2026 -- a period the model has genuinely never seen, nine years
later. That's the realistic scenario a grid operator actually faces:
"I have last year's pattern, how well can I predict a future quarter?"
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from preprocess import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN

MODEL_PATH = "models/demand_model.pkl"
METRICS_PATH = "models/metrics.json"
IMPORTANCE_PATH = "models/feature_importance.json"
DASHBOARD_DATA_PATH = "models/dashboard_data.json"


TEST_HOURS = 30 * 24  # hold out the most recent 30 days as the test set


def time_based_split(df: pd.DataFrame):
    """Chronological split: train on everything up to a cutoff, test on the
    most recent TEST_HOURS hours the model has never seen.

    This mirrors how the model would actually be used in production: you
    always train on the past and forecast the near future, never the other
    way round. A random shuffle-split would leak information (the model
    could "peek" at 2pm and 4pm while being tested on 3pm of the same day),
    so every point in the test set here comes strictly after every point in
    the training set.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = len(df_sorted) - TEST_HOURS
    train = df_sorted.iloc[:cutoff]
    test = df_sorted.iloc[cutoff:]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
    }


def main():
    # Make sure the output folder exists (empty folders are not stored by Git).
    os.makedirs("models", exist_ok=True)
    df = load_dataset()
    train_df, test_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    # --- Baseline: Linear Regression -------------------------------------
    # A simple, fast, interpretable baseline. If a complex model can't beat
    # this by a meaningful margin, the complexity isn't earning its keep.
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    linreg_metrics = evaluate(y_test.values, linreg.predict(X_test))

    # --- Main model: Random Forest Regressor ------------------------------
    # Chosen because demand depends on *non-linear interactions* between
    # calendar features (e.g. "weekday AND evening AND winter" behaves very
    # differently from any one of those alone). A Random Forest captures
    # these interactions automatically without manual feature crosses, is
    # robust to outliers/scale (no feature scaling needed), and gives us
    # feature importances for free -- useful for explaining the model.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred_test = rf.predict(X_test)
    rf_metrics = evaluate(y_test.values, rf_pred_test)

    print("Linear Regression (baseline):", linreg_metrics)
    print("Random Forest (final model): ", rf_metrics)

    # Persist the winning model
    joblib.dump(rf, MODEL_PATH)

    # --- Metrics + model comparison table ---------------------------------
    metrics_out = {
        "train_period": f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()} "
                         f"({len(train_df):,} hours)",
        "test_period": f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()} "
                        f"({len(test_df):,} hours, unseen -- most recent 30 days)",
        "models": {
            "linear_regression": linreg_metrics,
            "random_forest": rf_metrics,
        },
        "chosen_model": "random_forest",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    # --- Feature importance --------------------------------------------
    importance = dict(zip(FEATURE_COLUMNS, rf.feature_importances_.round(4)))
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
    with open(IMPORTANCE_PATH, "w") as f:
        json.dump({k: float(v) for k, v in importance.items()}, f, indent=2)

    # --- Pre-aggregated series for the dashboard's charts ------------------
    dashboard = {}

    # 1) Full time series (both years), downsampled to daily mean so the
    #    payload stays small and the line chart renders instantly.
    daily = df.set_index("timestamp")["demand_mw"].resample("D").mean().dropna()
    dashboard["daily_series"] = {
        "labels": [d.strftime("%Y-%m-%d") for d in daily.index],
        "values": daily.round(1).tolist(),
    }

    # 2) Average demand by hour of day (the classic "duck curve" shape)
    hourly_profile = df.groupby("hour")["demand_mw"].mean().round(1)
    dashboard["hourly_profile"] = {
        "labels": [f"{h:02d}:00" for h in hourly_profile.index],
        "values": hourly_profile.tolist(),
    }

    # 3) Average demand by month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_profile = df.groupby("month")["demand_mw"].mean().round(1)
    dashboard["monthly_profile"] = {
        "labels": [month_names[m - 1] for m in monthly_profile.index],
        "values": monthly_profile.tolist(),
    }

    # 4) Average demand by day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_profile = df.groupby("dayofweek")["demand_mw"].mean().round(1)
    dashboard["dayofweek_profile"] = {
        "labels": [dow_names[d] for d in dow_profile.index],
        "values": dow_profile.tolist(),
    }

    # 5) Actual vs Predicted on the held-out test set, downsampled to a
    #    daily mean for a readable chart (2,160 hourly points is too dense).
    test_df = test_df.copy()
    test_df["predicted"] = rf_pred_test
    daily_test = (
        test_df.set_index("timestamp")[["demand_mw", "predicted"]]
        .resample("D").mean().dropna()
    )
    dashboard["actual_vs_predicted"] = {
        "labels": [d.strftime("%Y-%m-%d") for d in daily_test.index],
        "actual": daily_test["demand_mw"].round(1).tolist(),
        "predicted": daily_test["predicted"].round(1).tolist(),
    }

    # 6) Summary stat cards
    dashboard["summary"] = {
        "peak_mw": round(float(df["demand_mw"].max()), 1),
        "min_mw": round(float(df["demand_mw"].min()), 1),
        "avg_mw": round(float(df["demand_mw"].mean()), 1),
        "total_hours": int(len(df)),
        "years_covered": sorted(df["year"].unique().tolist()),
        # The web app needs this to compute `trend_days` for a live
        # prediction exactly the same way training did (days since the
        # very first timestamp in the dataset).
        "dataset_start": df["timestamp"].min().strftime("%Y-%m-%d"),
        "dataset_end": df["timestamp"].max().strftime("%Y-%m-%d"),
    }

    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(dashboard, f)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved feature importance -> {IMPORTANCE_PATH}")
    print(f"Saved dashboard data -> {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    main()
