"""Loading the data and building model features."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "seattle-weather.csv"
WEATHER = ["drizzle", "fog", "rain", "snow", "sun"]
MEASUREMENTS = ["precipitation", "temp_max", "temp_min", "wind"]


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    missing = set(MEASUREMENTS + ["weather"]) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    return df


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """Day of year as a point on a circle, so 31 December sits next to 1 January."""
    df = df.copy()
    angle = 2 * np.pi * df["date"].dt.dayofyear / 365.25
    df["season_sin"] = np.sin(angle)
    df["season_cos"] = np.cos(angle)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    return df


TODAY_FEATURES = MEASUREMENTS + ["season_sin", "season_cos", "temp_range"]


def forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per day: what we know by the end of that day, and whether it rains tomorrow.

    Every feature uses only today or earlier, so the model never sees the future.
    """
    df = add_season(df)
    df["rain_today"] = (df["precipitation"] > 0).astype(int)
    for lag in (1, 2):
        for col in MEASUREMENTS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df["precip_3d"] = df["precipitation"].rolling(3).sum()
    df["precip_7d"] = df["precipitation"].rolling(7).sum()
    df["rain_days_7d"] = df["rain_today"].rolling(7).sum()
    df["rain_tomorrow"] = df["rain_today"].shift(-1)
    return df.dropna().astype({"rain_tomorrow": int}).reset_index(drop=True)


FORECAST_FEATURES = (TODAY_FEATURES + ["rain_today"]
                     + [f"{c}_lag{k}" for k in (1, 2) for c in MEASUREMENTS]
                     + ["precip_3d", "precip_7d", "rain_days_7d"])
