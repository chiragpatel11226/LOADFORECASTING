
"""
preprocess.py
-------------
Loads the raw NITI Aayog hourly power demand spreadsheet and turns it into a
clean, feature-engineered pandas DataFrame that the training script and the
Flask backend both reuse.

Why a separate module?
Keeping data-loading logic in one place means the training script and the
live web app can never disagree about how a "hour of day" or "is_weekend"
flag is computed. That consistency matters a lot for a time-series model:
if training features are built differently from inference-time features,
the model silently degrades.
"""

import re
import numpy as np
import pandas as pd

RAW_FILE = "data/Yearly_Demand_Profile.xlsx"

# The source file writes hours as "12am", "1am", ... "11pm", "12pm", ...
# Python's %I/%p format expects "12 AM" style tokens, so we normalise first.
_HOUR_PATTERN = re.compile(r"(\d{1,2})(am|pm)", re.IGNORECASE)


def _parse_hour_label(day_month: str, hour_label: str, year: int) -> pd.Timestamp:
    """Convert e.g. ('01-Jan', '12am', 2017) -> Timestamp('2017-01-01 00:00')."""
    match = _HOUR_PATTERN.search(hour_label.strip())
    hour_num, meridiem = int(match.group(1)), match.group(2).lower()

    if meridiem == "am":
        hour_24 = 0 if hour_num == 12 else hour_num
    else:  # pm
        hour_24 = 12 if hour_num == 12 else hour_num + 12

    # day_month looks like "01-Jan"; pandas can parse that together with year.
    base_date = pd.to_datetime(f"{day_month}-{year}", format="%d-%b-%Y")
    return base_date + pd.Timedelta(hours=hour_24)


def load_raw(path: str = RAW_FILE) -> pd.DataFrame:
    """Read the Excel export and drop the footer / metadata rows NITI Aayog
    appends after the real data (copyright notice, source URL, disclaimer)."""
    df = pd.read_excel(path)

    # Real rows always have a 4-digit numeric year. Footer rows have text
    # like "Copyright (c) 2026, NITI Aayog" in that same column, so this
    # single check cleanly separates data from metadata.
    df = df[df["Year"].astype(str).str.match(r"^\d{4}$")].copy()
    df["Year"] = df["Year"].astype(int)
    return df.reset_index(drop=True)


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Split the 'Date' column (e.g. '01-Jan 12am') into a real day-month
    string and hour label, then construct a proper pandas datetime index."""
    df = df.copy()
    split = df["Date"].str.extract(r"^(\d{2}-[A-Za-z]{3})\s+(\d{1,2}(?:am|pm))$")
    df["day_month"] = split[0]
    df["hour_label"] = split[1]

    df["timestamp"] = [
        _parse_hour_label(dm, hl, yr)
        for dm, hl, yr in zip(df["day_month"], df["hour_label"], df["Year"])
    ]

    df = df.rename(columns={"Hourly Demand Met (in MW)": "demand_mw"})
    df = df[["timestamp", "demand_mw"]].sort_values("timestamp").reset_index(drop=True)

    # A handful of duplicate timestamps can occur across raw exports; keep
    # the first occurrence and drop true duplicates to protect the model
    # from being trained on the same target twice.
    df = df.drop_duplicates(subset="timestamp", keep="first")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer purely calendar-based features (no leakage from the target).

    We deliberately avoid lag/rolling features (e.g. "demand 24h ago") here.
    Lag features usually *improve* accuracy for next-hour forecasting, but
    they require knowing recent real demand at prediction time. Since the
    web app lets a user ask "what would demand look like on a Tuesday in
    July at 3pm", there is no "recent history" available for an arbitrary
    hypothetical date. A calendar-only model trades a bit of accuracy for
    the ability to answer that kind of open-ended, what-if question -- and
    it's easy to defend that trade-off in an interview.
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["dayofweek"] = ts.dt.dayofweek  # 0=Monday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["day_of_year"] = ts.dt.dayofyear
    df["year"] = ts.dt.year

    # Cyclical (sine/cosine) encoding: hour 23 and hour 0 are one hour apart
    # in reality but would look 23 apart to a model using the raw integer.
    # Sin/cos encoding wraps the value around a circle so the model sees
    # them as neighbours, which matters a lot for a 24-hour daily cycle.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Trend feature: grid-wide power demand grows year over year as economies
    # grow (India's demand rose sharply between 2017 and 2026 in this data).
    # Pure seasonal features (hour/month/day-of-week) repeat identically every
    # year, so without an explicit trend term the model has no way to know
    # that "January 2026" typically runs much higher than "January 2017" --
    # it would systematically under-predict every recent value. `trend_days`
    # is simply the number of days since the dataset's first timestamp,
    # giving the model a numeric handle on long-term growth.
    df["trend_days"] = (ts - ts.min()).dt.total_seconds() / 86400

    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "day_of_year",
    "trend_days",
]
TARGET_COLUMN = "demand_mw"


def load_dataset(path: str = RAW_FILE) -> pd.DataFrame:
    """Convenience one-call pipeline: raw Excel -> clean, feature-rich df."""
    raw = load_raw(path)
    dated = build_datetime(raw)
    featured = add_calendar_features(dated)
    return featured


if __name__ == "__main__":
    data = load_dataset()
    print(data.shape)
    print(data.head())
    print(data["year"].value_counts())
    print(data[TARGET_COLUMN].describe())
