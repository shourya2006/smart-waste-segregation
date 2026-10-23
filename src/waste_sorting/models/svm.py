"""Training-only SVMs with internal selection, scaling and strict convergence."""

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC


def _positive_number(value):
    return (
        not isinstance(value, bool) and isinstance(value, (int, float)) and 0 < value < float("inf")
    )


class StrictLinearSVC(LinearSVC):
    """Reject unconverged fits inside joblib workers as well as in serial searches."""

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return super().fit(X, y, sample_weight=sample_weight)


def sqrt_features(X):
    """Per-image histogram root transform; never learns across images."""
    values = np.asarray(X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Root normalization requires finite nonnegative features")
    return np.sqrt(values)


def make_pipeline(model_name, columns, seed=42, normalization="standard", linear_max_iter=20000):
    """Accept full cached vectors, selecting the same columns at fit and prediction."""
    selection = ColumnTransformer([("selected", "passthrough", list(columns))])
    if model_name == "dummy":
        return Pipeline(
            [
                ("features", selection),
                ("classifier", DummyClassifier(strategy="most_frequent", random_state=seed)),
            ]
        )
    if normalization == "standard":
        scale = StandardScaler()
    elif normalization == "root_l2":
        scale = Pipeline(
            [
                ("root", FunctionTransformer(sqrt_features, validate=True)),
                ("normalize", Normalizer(norm="l2")),
            ]
        )
    else:
        raise ValueError("normalization must be 'standard' or 'root_l2'")
    if model_name == "linear_svm":
        classifier = StrictLinearSVC(dual="auto", max_iter=linear_max_iter, random_state=seed)
    elif model_name == "rbf_svm":
        classifier = SVC(kernel="rbf", cache_size=256, random_state=seed)
    else:
        raise ValueError(f"Unknown baseline: {model_name!r}")
    return Pipeline(
        [
            ("features", selection),
            ("scale", scale),
            ("classifier", classifier),
        ]
    )


def parameter_grid(model_name, config):
    """Read a predeclared search; 'none' in TOML represents no class weighting."""
    if model_name == "dummy":
        return {}
    if model_name not in {"linear_svm", "rbf_svm"}:
        raise ValueError(f"Unknown baseline: {model_name!r}")
    settings = config["training"][model_name]
    weights = settings["class_weight"]
    if not weights or any(w not in {"none", "balanced"} for w in weights):
        raise ValueError("class_weight search must contain 'none' and/or 'balanced'")
    values = settings["C"]
    if not values or not all(_positive_number(v) for v in values):
        raise ValueError("C search values must be finite positive numbers")
    grid = {
        "classifier__C": values,
        "classifier__class_weight": [None if w == "none" else w for w in weights],
    }
    if model_name == "rbf_svm":
        gamma = settings["gamma"]
        if not gamma or not all(
            v in {"scale", "auto"} if isinstance(v, str) else _positive_number(v) for v in gamma
        ):
            raise ValueError("gamma must contain 'scale', 'auto', or finite positive numbers")
        grid["classifier__gamma"] = gamma
    return grid
