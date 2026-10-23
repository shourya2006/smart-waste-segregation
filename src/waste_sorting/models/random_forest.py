"""Random Forest pipelines and parameter grid definitions.

Input feature vectors are kept full (1,806 dimensions), and descriptor subsets
are selected inside the scikit-learn Pipeline via ColumnTransformer.
Random Forest does not require feature scaling.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def _positive_int(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 < value < float("inf")
    )


def make_pipeline(columns, seed=42, n_jobs=None):
    """Build a Random Forest pipeline selecting specified columns.

    Random Forest is invariant to monotonic feature scaling, so no StandardScaler
    is included in this pipeline.
    """
    col_list = list(columns)
    if not col_list:
        raise ValueError("Must specify at least one feature column")
    selection = ColumnTransformer([("selected", "passthrough", col_list)])
    classifier = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    return Pipeline(
        [
            ("features", selection),
            ("classifier", classifier),
        ]
    )


def parameter_grid(config):
    """Read a predeclared Random Forest search grid from configuration.

    Supports 'none' and 'balanced' class_weight, n_estimators, max_depth,
    and min_samples_split choices.
    """
    rf_settings = config["training"].get("random_forest", {})
    if not rf_settings:
        raise ValueError("Configuration missing [training.random_forest] section")

    weights = rf_settings.get("class_weight", ["none", "balanced"])
    if not weights or any(w not in {"none", "balanced"} for w in weights):
        raise ValueError("class_weight search must contain 'none' and/or 'balanced'")

    n_estimators = rf_settings.get("n_estimators", [100, 200])
    if not n_estimators or not all(_positive_int(v) for v in n_estimators):
        raise ValueError("n_estimators search values must be positive integers")

    max_depth_raw = rf_settings.get("max_depth", ["none", 10, 20])
    if not max_depth_raw:
        raise ValueError("max_depth search values must not be empty")
    max_depth = []
    for d in max_depth_raw:
        if d in {"none", None}:
            max_depth.append(None)
        elif _positive_int(d):
            max_depth.append(d)
        else:
            raise ValueError(f"Invalid max_depth value: {d!r}")

    min_samples_split = rf_settings.get("min_samples_split", [2, 5])
    if not min_samples_split or not all(_positive_int(v) and v >= 2 for v in min_samples_split):
        raise ValueError("min_samples_split search values must be integers >= 2")

    return {
        "classifier__n_estimators": n_estimators,
        "classifier__max_depth": max_depth,
        "classifier__min_samples_split": min_samples_split,
        "classifier__class_weight": [None if w == "none" else w for w in weights],
    }
