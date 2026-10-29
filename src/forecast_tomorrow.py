"""Part 2: a real forecast. Will it rain tomorrow?

Uses only what is known by the end of today. Trains on 2012-2014 and tests on
2015, so the model is always predicting days that come after everything it
learned from, just as a real forecast would.

    python src/forecast_tomorrow.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import FORECAST_FEATURES, ROOT, forecast_table, load  # noqa: E402
from models import random_forest, xgboost  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
TEST_YEAR = 2015


def time_split(table: pd.DataFrame, test_year: int = TEST_YEAR):
    train = table[table["date"].dt.year < test_year]
    test = table[table["date"].dt.year == test_year]
    return train, test


def score(name, y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, prob) if pd.Series(prob).nunique() > 2 else None  # needs a range of scores
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, pred), 3),
            "f1_rain": round(f1_score(y_true, pred), 3),
            "roc_auc": round(auc, 3) if auc is not None else None,
            "brier": round(brier_score_loss(y_true, prob), 3)}


def main() -> pd.DataFrame:
    table = forecast_table(load())
    train, test = time_split(table)
    y_tr, y_te = train["rain_tomorrow"], test["rain_tomorrow"]
    rows = [
        score("Always 'no rain'", y_te, pd.Series(0.0, index=y_te.index)),
        score("Persistence: tomorrow = today", y_te, test["rain_today"].astype(float)),
        score("Climatology: rain rate for that month",
              y_te, test["date"].dt.month.map(train.groupby(train["date"].dt.month)["rain_tomorrow"].mean())),
    ]
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": random_forest(),
    }
    xg = xgboost(n_estimators=300, max_depth=3, learning_rate=0.05)
    if xg is not None:
        models["XGBoost"] = xg
    for name, model in models.items():
        model.fit(train[FORECAST_FEATURES], y_tr)
        rows.append(score(name, y_te, model.predict_proba(test[FORECAST_FEATURES])[:, 1]))

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "forecast_results.csv", index=False)
    plot(results)
    print(f"Train {len(train)} days (2012-2014), test {len(test)} days ({TEST_YEAR}); "
          f"it rained on {y_te.mean():.0%} of test days")
    print(results.to_string(index=False))
    return results


def plot(results):
    r = results.iloc[::-1]
    baselines = ("Always", "Persistence", "Climatology")
    colors = ["#9aa6b5" if m.startswith(baselines) else "#1f5f8b" for m in r["model"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(r["model"], r["accuracy"], color=colors)
    for y, v in enumerate(r["accuracy"]):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0.5, max(r["accuracy"].max() + 0.06, 0.8))
    ax.set_xlabel(f"Accuracy on {TEST_YEAR}")
    ax.set_title("Will it rain tomorrow? Models vs simple baselines (grey)", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "forecast_accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
