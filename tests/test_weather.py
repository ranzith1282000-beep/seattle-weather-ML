"""Checks that guard against the classic ways a model can look better than it is."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classify_today import precipitation_rule  # noqa: E402
from features import FORECAST_FEATURES, forecast_table, load  # noqa: E402
from forecast_tomorrow import time_split  # noqa: E402


def test_data_loads_cleanly():
    df = load()
    assert len(df) == 1461
    assert df.isna().sum().sum() == 0
    assert df["date"].is_monotonic_increasing


def test_target_is_tomorrows_rain():
    t = forecast_table(load())
    # rain_tomorrow on day d must equal rain_today on day d+1
    nxt = t.set_index("date")["rain_today"].shift(-1, freq="D")
    joined = t.set_index("date")[["rain_tomorrow"]].join(nxt.rename("actual_next"), how="inner")
    assert (joined["rain_tomorrow"] == joined["actual_next"]).all()


def test_features_never_use_the_future():
    df = load()
    base = forecast_table(df)
    # Change every value from 1 July 2014 onwards: features for earlier days must not move.
    altered = df.copy()
    future = altered["date"] >= "2014-07-01"
    altered.loc[future, ["precipitation", "temp_max", "temp_min", "wind"]] *= 3
    changed = forecast_table(altered)
    before = base["date"] < "2014-07-01"
    pd.testing.assert_frame_equal(base.loc[before, FORECAST_FEATURES], changed.loc[before, FORECAST_FEATURES])


def test_time_split_has_no_overlap():
    train, test = time_split(forecast_table(load()))
    assert train["date"].max() < test["date"].min()
    assert set(test["date"].dt.year) == {2015}


def test_precipitation_rule():
    X = pd.DataFrame({"precipitation": [0.0, 5.0, 3.0], "temp_max": [20.0, 12.0, 2.0]})
    assert list(precipitation_rule(X)) == ["sun", "rain", "snow"]
