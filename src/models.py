"""Shared model helpers. XGBoost is used when installed; scikit-learn models always run."""

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # keeps the rest of the project runnable without XGBoost
    XGBClassifier = None


def random_forest(balanced: bool = False):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0,
                                  class_weight="balanced" if balanced else None, n_jobs=-1)


def xgboost(**params):
    if XGBClassifier is None:
        return None
    return XGBClassifier(random_state=0, n_jobs=1, eval_metric="mlogloss", **params)
