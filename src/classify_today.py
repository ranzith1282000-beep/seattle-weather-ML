"""Part 1: classify each day's weather type from that day's measurements.

Compares the original XGBoost model with simple baselines, using the same
random 80/20 split as the original notebook so the numbers are comparable.

    python src/classify_today.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from features import MEASUREMENTS, ROOT, TODAY_FEATURES, WEATHER, add_season, load  # noqa: E402
from models import random_forest, xgboost  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def precipitation_rule(X: pd.DataFrame) -> np.ndarray:
    """No machine learning: wet days are rain (snow if it stayed near freezing), dry days are sun."""
    wet = X["precipitation"] > 0
    return np.where(wet, np.where(X["temp_max"] <= 4, "snow", "rain"), "sun")


def score(name: str, y_true, y_pred) -> dict:
    recalls = recall_score(y_true, y_pred, labels=WEATHER, average=None, zero_division=0)
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, labels=WEATHER, average="macro", zero_division=0), 3),
            **{f"recall_{w}": round(r, 2) for w, r in zip(WEATHER, recalls)}}


def main() -> pd.DataFrame:
    df = add_season(load())
    y = df["weather"]
    codes = {w: i for i, w in enumerate(WEATHER)}  # same alphabetical codes as LabelEncoder
    names = np.array(WEATHER)

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    tr, te = df.loc[train_idx], df.loc[test_idx]
    rows = []

    rows.append(score("Always predict the most common type", te["weather"],
                      np.full(len(te), tr["weather"].mode()[0])))
    rows.append(score("Precipitation rule (no ML)", te["weather"], precipitation_rule(te)))

    xg = xgboost()
    if xg is not None:
        # The original notebook: 4 measurements, default settings, then a grid search on accuracy.
        xg.fit(tr[MEASUREMENTS], tr["weather"].map(codes))
        rows.append(score("XGBoost, original settings", te["weather"], names[xg.predict(te[MEASUREMENTS])]))

        grid = GridSearchCV(xgboost(), {"learning_rate": [0.1, 1, 0.01, 0.001], "gamma": [0, 1, 10, 100]},
                            cv=10, scoring="accuracy")
        grid.fit(tr[MEASUREMENTS], tr["weather"].map(codes))
        rows.append(score("XGBoost, original grid search", te["weather"],
                          names[grid.predict(te[MEASUREMENTS])]))

        # Improved: season features, rare types weighted up, tuned on macro F1 instead of accuracy.
        tuned = GridSearchCV(xgboost(), {"learning_rate": [0.05, 0.1], "max_depth": [3, 5],
                                         "n_estimators": [200, 400]},
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro")
        tuned.fit(tr[TODAY_FEATURES], tr["weather"].map(codes),
                  sample_weight=compute_sample_weight("balanced", tr["weather"]))
        rows.append(score("XGBoost, season features + balanced weights", te["weather"],
                          names[tuned.predict(te[TODAY_FEATURES])]))
    else:
        print("XGBoost not installed: skipping the XGBoost rows (pip install xgboost)")

    rf = random_forest(balanced=True).fit(tr[TODAY_FEATURES], tr["weather"])
    rf_pred = rf.predict(te[TODAY_FEATURES])
    rows.append(score("Random forest, season features + balanced weights", te["weather"], rf_pred))

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "today_results.csv", index=False)
    (RESULTS / "test_set_counts.json").write_text(json.dumps(te["weather"].value_counts().to_dict(), indent=2))
    plot_confusion(te["weather"], rf_pred, "random forest, season features + balanced weights")
    plot_labels_by_year(df)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.to_string(index=False))
    return results


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=WEATHER)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), WEATHER)
    ax.set_yticks(range(5), WEATHER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2a37")
    ax.set_title(f"Test set: {title}", fontsize=10, loc="left")
    fig.tight_layout()
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_labels_by_year(df):
    counts = pd.crosstab(df["date"].dt.year, df["weather"])[WEATHER]
    colors = {"drizzle": "#8fb8d8", "fog": "#9aa6b5", "rain": "#1f5f8b", "snow": "#c9d6e8", "sun": "#e0a43b"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.15
    years = counts.index.to_numpy()
    for k, w in enumerate(WEATHER):
        ax.bar(years + (k - 2) * width, counts[w], width, label=w, color=colors[w])
    ax.set_xticks(years)
    ax.set_ylabel("Days")
    ax.set_title("Days labelled with each weather type, by year", loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "labels_by_year.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
