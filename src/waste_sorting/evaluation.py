"""Group-aware calibration, validation abstention selection, and frozen test evaluation.

Phase 1 deliverable for Prajjwal Tripathi (Confidence Calibration and Evaluation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from waste_sorting.acquisition import sha256
from waste_sorting.extended_features import (
    DIMENSION as EXTENDED_DIMENSION,
)
from waste_sorting.extended_features import (
    SETTINGS as EXTENDED_SETTINGS,
)
from waste_sorting.extended_features import (
    extended_columns,
    extract_extended_features,
)
from waste_sorting.models.svm import make_pipeline
from waste_sorting.training import (
    _source_metadata,
    _validate_cv,
    load_or_create_folds,
    load_training_data,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SELECTED_FAMILY = "global_color_spatial_texture"
SELECTED_DIMENSION = 1608
HYPERPARAMETERS = {
    "C": 10.0,
    "gamma": 10.0,
    "class_weight": None,
    "seed": 42,
}


def load_manifest(root: Path) -> pd.DataFrame:
    """Load and validate manifest for image IDs, splits, labels and duplicate groups."""
    manifest_path = root / "docs/dataset/manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"image_id", "label", "included", "split", "duplicate_group", "sha256"}
    if not required.issubset(manifest.columns):
        raise ValueError("Manifest missing required identity/split columns")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs")

    included = manifest[manifest["included"] == True].sort_values("image_id").copy()  # noqa: E712
    if set(included["split"]) != {"train", "validation", "test"}:
        raise ValueError("All three outer splits (train, validation, test) must be present")
    if (included.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Duplicate group crosses outer split boundaries")
    if not included["label"].isin(CLASSES).all():
        raise ValueError(f"Manifest contains classes outside configured {CLASSES}")
    return included


def feature_cache_provenance(root: Path) -> dict[str, str]:
    """Fingerprint all local source and settings used to produce extended features."""
    root = Path(root)
    return {
        "manifest_sha256": sha256(root / "docs/dataset/manifest.csv"),
        "extractor_sha256": sha256(root / "src/waste_sorting/extended_features.py"),
        "preprocessing_sha256": sha256(root / "src/waste_sorting/preprocessing.py"),
        "settings_hash": hashlib.sha256(
            json.dumps(EXTENDED_SETTINGS, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def validate_feature_cache(
    cache_path: str | Path,
    split_name: str,
    root: Path,
    manifest: pd.DataFrame | None = None,
) -> np.ndarray:
    """Validate identity, extractor/preprocessing provenance, and integrity of a feature cache.

    Raises:
        FileNotFoundError: If cache file does not exist.
        ValueError: If cache is malformed, unverifiable (missing provenance keys),
            stale (mismatched manifest, extractor, or settings hash), wrong split,
            wrong image IDs, wrong dtype, wrong shape, or contains non-finite or negative values.
    """
    cache_file = Path(cache_path)
    if not cache_file.exists():
        raise FileNotFoundError(f"Feature cache file not found: {cache_file}")

    try:
        data = np.load(cache_file, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Malformed cache file {cache_file}: {exc}") from exc

    required_keys = {
        "features",
        "image_ids",
        "split",
        "manifest_sha256",
        "extractor_sha256",
        "preprocessing_sha256",
        "settings_hash",
    }
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(
            f"Unverifiable feature cache at {cache_file}: missing required provenance keys: "
            f"{sorted(missing)}"
        )

    # Validate split identity
    cached_split = str(data["split"])
    if cached_split != split_name:
        raise ValueError(f"Cache split mismatch: expected {split_name!r}, found {cached_split!r}")

    # The writer and reader share the same complete provenance contract.
    expected = feature_cache_provenance(root)
    descriptions = {
        "manifest_sha256": "manifest checksum",
        "extractor_sha256": "extractor checksum",
        "preprocessing_sha256": "preprocessing checksum",
        "settings_hash": "feature extraction settings hash",
    }
    for key, expected_value in expected.items():
        cached_value = str(data[key])
        if cached_value != expected_value:
            raise ValueError(
                f"Stale feature cache: {descriptions[key]} mismatch "
                f"(cache has {cached_value!r}, current value is {expected_value!r})"
            )

    # Validate image IDs against manifest
    if manifest is None:
        manifest = load_manifest(root)
    split_df = manifest[manifest["split"] == split_name].sort_values("image_id")
    expected_ids = split_df["image_id"].to_numpy()
    cached_ids = data["image_ids"]
    if not np.array_equal(cached_ids, expected_ids):
        raise ValueError(f"Cache image IDs do not match manifest split {split_name!r} image IDs")

    # Validate numeric dtype
    features = data["features"]
    if not isinstance(features, np.ndarray) or features.dtype != np.float32:
        dtype_str = getattr(features, "dtype", type(features))
        raise ValueError(f"Invalid feature matrix dtype: expected float32, got {dtype_str}")

    # Validate dimensions
    expected_shape = (len(expected_ids), EXTENDED_DIMENSION)
    if features.shape != expected_shape:
        raise ValueError(
            f"Invalid feature matrix shape: expected {expected_shape}, got {features.shape}"
        )

    # Validate finite values
    if not np.isfinite(features).all():
        raise ValueError("Feature cache contains non-finite values (NaN or Inf)")

    # Validate nonnegative values
    if (features < 0.0).any():
        raise ValueError(
            "Feature cache contains negative values; extended histograms must be nonnegative"
        )

    return features


def extract_split_features(
    split_name: str,
    root: Path,
    manifest: pd.DataFrame | None = None,
    *,
    n_jobs: int = -1,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract extended 2,760-dim feature matrix for an outer split.

    Verifies every image file's SHA256 checksum against the manifest.
    Optionally reads from/saves to a local cache in the ignored artifacts directory.
    Validates manifest, split, extractor, and preprocessing provenance before reusing cache.
    """
    root = Path(root).resolve()
    if manifest is None:
        manifest = load_manifest(root)

    split_df = manifest[manifest["split"] == split_name].sort_values("image_id").copy()
    if split_df.empty:
        raise ValueError(f"Split {split_name!r} has no included images")

    expected_ids = split_df["image_id"].to_numpy()
    target_map = {name: i for i, name in enumerate(CLASSES)}
    expected_y = split_df["label"].map(target_map).to_numpy(dtype=np.int64)
    expected_groups = split_df["duplicate_group"].to_numpy()

    if cache_path and Path(cache_path).exists():
        cached_matrix = validate_feature_cache(
            cache_path=cache_path,
            split_name=split_name,
            root=root,
            manifest=manifest,
        )
        return cached_matrix, expected_y, expected_ids, expected_groups

    raw_dir = root / "data/raw/trashnet"

    def extract_single(row):
        img_path = raw_dir / row.image_id
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image file: {img_path}")
        if sha256(img_path) != row.sha256:
            raise ValueError(f"Image checksum mismatch for {row.image_id}")
        with Image.open(img_path) as opened:
            return extract_extended_features(opened)

    rows = list(split_df.itertuples())
    extracted = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(extract_single)(r) for r in rows)
    matrix = np.asarray(extracted, dtype=np.float32)

    if matrix.shape != (len(split_df), EXTENDED_DIMENSION) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid feature matrix for split {split_name}")
    if (matrix < 0.0).any():
        raise ValueError("Extended histograms must be nonnegative")

    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=matrix,
            image_ids=expected_ids,
            split=split_name,
            **feature_cache_provenance(root),
        )

    return matrix, expected_y, expected_ids, expected_groups


def reconstruct_selected_pipeline(seed: int = 42):
    """Reconstruct the joint-study winning extended RBF SVM pipeline.

    Selected configuration:
      - Feature family: global_color_spatial_texture (1,608 columns selected from 2,760 input)
      - Normalization: elementwise sqrt then per-row L2 (root_l2)
      - Classifier: RBF SVC with C=10.0, gamma=10.0, class_weight=None, seed=42
    """
    pipe = make_pipeline(
        "rbf_svm",
        extended_columns(SELECTED_FAMILY),
        normalization="root_l2",
        seed=seed,
    )
    pipe.set_params(
        classifier__C=HYPERPARAMETERS["C"],
        classifier__gamma=HYPERPARAMETERS["gamma"],
        classifier__class_weight=HYPERPARAMETERS["class_weight"],
    )
    return pipe


def fit_calibrated_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 42,
    n_jobs: int = 1,
) -> CalibratedClassifierCV:
    """Fit group-aware sigmoid-calibrated ensemble on outer training data only.

    For each fold k:
      - The base estimator is fitted only on fold k's training complement.
      - The Platt sigmoid calibrator is fitted on fold k's withheld training rows.
      - Never calibrates on rows used for base fitting.
      - Preserves ensemble=True so the deployed predictor is the calibrated ensemble.
    """
    if X_train.shape[1] != EXTENDED_DIMENSION:
        raise ValueError(
            f"Expected {EXTENDED_DIMENSION} input features, got {X_train.shape[1]}. "
            "Pipeline selects 1,608 internal columns."
        )
    if len(folds) != 5:
        raise ValueError(f"Expected 5 explicit frozen folds, got {len(folds)}")

    base_pipeline = reconstruct_selected_pipeline(seed=seed)
    calibrator = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=folds,
        ensemble=True,
        n_jobs=n_jobs,
    )
    calibrator.fit(X_train, y_train)

    if len(calibrator.calibrated_classifiers_) != len(folds):
        raise ValueError("CalibratedClassifierCV did not produce 5 ensemble members")
    if not np.array_equal(calibrator.classes_, np.arange(len(CLASSES))):
        raise ValueError("Calibrator class indices do not match expected range")

    # Verification on sample rows
    sample_probs = calibrator.predict_proba(X_train[:10])
    if sample_probs.shape != (10, len(CLASSES)):
        raise ValueError("Probability matrix shape mismatch")
    if not np.isfinite(sample_probs).all() or (sample_probs < 0.0).any():
        raise ValueError("Non-finite or negative probabilities produced")
    if not np.allclose(sample_probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probabilities do not sum to 1.0")

    return calibrator


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 6) -> float:
    """Calculate Multiclass Brier score.

    Formula:
        BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2
    where y_{ic} is the one-hot indicator of true class membership and p_{ic}
    is the predicted calibrated probability for class c.
    Lower is better, ranging from 0.0 (perfect prediction) to 2.0 (maximum error).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(f"Expected y_prob of shape (N, {n_classes}), got {y_prob.shape}")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("Length mismatch between y_true and y_prob")
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_calibration_curve_and_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute 10-bin confidence reliability diagram statistics and Expected Calibration Error.

    Bins are 10 fixed confidence intervals:
        [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    where the final bin includes the upper boundary 1.0.

    For each bin B_m:
        - acc(B_m): fraction of top-1 predictions in B_m that are correct
        - conf(B_m): mean top-1 predicted confidence in B_m
        - count: number of samples in B_m

    Expected Calibration Error (ECE):
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    where empty bins (|B_m| == 0) contribute 0 to the sum.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correctness = (predictions == y_true).astype(float)
    total_samples = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    ece = 0.0

    for idx in range(n_bins):
        low, high = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(correctness[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((low + high) / 2.0)

        bin_details.append(
            {
                "bin_index": idx,
                "range": [round(float(low), 2), round(float(high), 2)],
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
            }
        )

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bins": bin_details,
    }


def sweep_abstention_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Sweep threshold grid 0.00 through 1.00 in increments of 0.01.

    Rule: Accept when max calibrated probability is >= threshold.
    Coverage = accepted examples / all examples.
    Accepted accuracy = correct accepted predictions / accepted examples.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    is_correct = predictions == y_true
    n_total = len(y_true)

    records = []
    # Grid 0.00 to 1.00 inclusive in increments of 0.01 (101 thresholds)
    for step in range(101):
        threshold = round(step * 0.01, 2)
        accepted_mask = confidences >= threshold
        n_accepted = int(np.sum(accepted_mask))
        n_abstained = n_total - n_accepted
        coverage = float(n_accepted / n_total)
        if n_accepted > 0:
            acc = float(np.mean(is_correct[accepted_mask]))
        else:
            acc = 0.0

        records.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accepted_accuracy": acc,
                "accepted_count": n_accepted,
                "abstained_count": n_abstained,
                "total_count": n_total,
            }
        )

    return pd.DataFrame(records)


def select_abstention_threshold(
    sweep_df: pd.DataFrame,
    target_accuracy: float = 0.90,
    min_coverage: float = 0.80,
) -> dict[str, Any]:
    """Select optimal threshold using validation data only.

    Target criteria:
      1. Accepted accuracy >= 90% (0.90)
      2. Coverage >= 80% (0.80)
      Selection: threshold with highest coverage satisfying both.
      Tie-breaking: resolve ties using the lower threshold.

    Fallback (if no threshold satisfies both):
      Selection: highest accepted accuracy among thresholds with coverage >= 80%.
      Tie-breaking: higher coverage, then lower threshold.
      Condition: report clearly that the requested accuracy condition was not achieved.
    """
    feasible = sweep_df[
        (sweep_df["accepted_accuracy"] >= target_accuracy) & (sweep_df["coverage"] >= min_coverage)
    ]

    if not feasible.empty:
        max_coverage = feasible["coverage"].max()
        tied = feasible[np.isclose(feasible["coverage"], max_coverage)]
        # Lowest threshold resolves ties
        selected_row = tied.sort_values("threshold", ascending=True).iloc[0]
        return {
            "threshold": float(selected_row["threshold"]),
            "coverage": float(selected_row["coverage"]),
            "accepted_accuracy": float(selected_row["accepted_accuracy"]),
            "accepted_count": int(selected_row["accepted_count"]),
            "abstained_count": int(selected_row["abstained_count"]),
            "total_count": int(selected_row["total_count"]),
            "target_met": True,
            "target_accuracy": target_accuracy,
            "min_coverage": min_coverage,
            "selection_rule": (
                "highest coverage with accuracy >= target and coverage >= min_coverage; "
                "ties to lower threshold"
            ),
        }

    # Infeasible: fallback
    eligible = sweep_df[sweep_df["coverage"] >= min_coverage]
    if eligible.empty:
        raise ValueError(f"No threshold achieved minimum coverage of {min_coverage}")

    max_acc = eligible["accepted_accuracy"].max()
    tied_acc = eligible[np.isclose(eligible["accepted_accuracy"], max_acc)]
    # Higher coverage, then lower threshold
    selected_row = tied_acc.sort_values(by=["coverage", "threshold"], ascending=[False, True]).iloc[
        0
    ]

    return {
        "threshold": float(selected_row["threshold"]),
        "coverage": float(selected_row["coverage"]),
        "accepted_accuracy": float(selected_row["accepted_accuracy"]),
        "accepted_count": int(selected_row["accepted_count"]),
        "abstained_count": int(selected_row["abstained_count"]),
        "total_count": int(selected_row["total_count"]),
        "target_met": False,
        "target_accuracy": target_accuracy,
        "min_coverage": min_coverage,
        "selection_rule": (
            "fallback: highest accepted accuracy among coverage >= min_coverage; "
            "ties to higher coverage, then lower threshold"
        ),
        "note": (
            f"Target accuracy >= {target_accuracy:.0%} was not achieved on validation "
            f"with coverage >= {min_coverage:.0%}."
        ),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive unconditional and thresholded selective metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    n_samples = len(y_true)

    unconditional_acc = float(accuracy_score(y_true, predictions))
    unconditional_macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, predictions, labels=list(range(len(class_names))))
    brier = multiclass_brier_score(y_true, y_prob, n_classes=len(class_names))
    cross_entropy = float(log_loss(y_true, y_prob, labels=list(range(len(class_names)))))
    cal_info = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=10)

    result: dict[str, Any] = {
        "sample_count": n_samples,
        "classes": list(class_names),
        "unconditional": {
            "accuracy": unconditional_acc,
            "macro_f1": unconditional_macro_f1,
            "log_loss": cross_entropy,
            "log_loss_formula": "- (1 / N) * sum_{i=1}^N sum_{c=1}^C y_{ic} * ln(p_{ic})",
            "brier_score": brier,
            "brier_score_formula": "(1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2",
            "expected_calibration_error": cal_info["ece"],
            "ece_formula": "sum_{m=1}^{10} (|B_m| / N) * |acc(B_m) - conf(B_m)|",
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "calibration_bins": cal_info["bins"],
        },
    }

    if threshold is not None:
        accepted_mask = confidences >= threshold
        n_accepted = int(np.sum(accepted_mask))
        n_abstained = n_samples - n_accepted
        coverage = float(n_accepted / n_samples)
        if n_accepted > 0:
            accepted_acc = float(accuracy_score(y_true[accepted_mask], predictions[accepted_mask]))
            accepted_f1 = float(
                f1_score(
                    y_true[accepted_mask],
                    predictions[accepted_mask],
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            accepted_acc = 0.0
            accepted_f1 = 0.0

        result["abstention"] = {
            "threshold": float(threshold),
            "coverage": coverage,
            "accepted_accuracy": accepted_acc,
            "accepted_macro_f1": accepted_f1,
            "accepted_count": n_accepted,
            "abstained_count": n_abstained,
            "total_count": n_samples,
        }

    return result


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    output_path: Path,
    title: str = "Test Confusion Matrix",
) -> None:
    """Save formatted confusion matrix plot."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_reliability_diagram(
    cal_info: dict[str, Any],
    output_path: Path,
    title: str = "Confidence Reliability Diagram (10 Bins)",
) -> None:
    """Save reliability diagram bar plot with identity reference."""
    bins = cal_info["bins"]
    bin_centers = [round((b["range"][0] + b["range"][1]) / 2.0, 2) for b in bins]
    bin_accuracies = [b["accuracy"] for b in bins]
    bin_confs = [b["confidence"] for b in bins]
    bin_counts = [b["count"] for b in bins]

    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Perfect calibration reference
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Accuracy bars vs confidence
    width = 0.07
    bars = ax1.bar(
        bin_centers,
        bin_accuracies,
        width=width,
        alpha=0.65,
        color="#2b5c8f",
        edgecolor="#1a3b5c",
        label="Bin Accuracy",
    )
    ax1.plot(bin_centers, bin_confs, "r.-", markersize=8, label="Bin Confidence")

    # Annotate counts
    for bar, count in zip(bars, bin_counts, strict=True):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("Confidence", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ece_val = cal_info["ece"]
    ax1.set_title(f"{title}\nExpected Calibration Error (ECE): {ece_val:.4f}", fontsize=12, pad=12)
    ax1.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_pipeline(
    root: Path,
    output_dir: Path,
    artifact_dir: Path,
    *,
    n_jobs: int = 1,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Execute complete calibration, validation abstention selection, and frozen test evaluation."""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve()

    test_pred_path = output_dir / "test-predictions.csv"
    test_metrics_path = output_dir / "test-metrics.json"

    # Guard against accidentally overwriting existing final test evaluation
    if (test_pred_path.exists() or test_metrics_path.exists()) and not allow_overwrite:
        raise FileExistsError(
            f"Final test evaluation outputs already exist in {output_dir}. "
            "A guard prevents accidentally overwriting an existing final evaluation. "
            "Pass allow_overwrite=True if intentional."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Validating manifest and loading frozen training folds...", flush=True)
    manifest = load_manifest(root)
    base_training = load_training_data(root / "data/processed/features.npz", root)
    fold_path = root / "docs/training/folds.json"
    folds = load_or_create_folds(base_training, fold_path)
    _validate_cv(base_training, folds)

    # Provenance tracking
    source_meta = _source_metadata(root)
    calibration_protocol = {
        "scope": "Phase 1 Calibration and Evaluation",
        "selected_model": "extended_rbf_svm",
        "feature_family": SELECTED_FAMILY,
        "input_feature_dimension": EXTENDED_DIMENSION,
        "selected_feature_dimension": SELECTED_DIMENSION,
        "hyperparameters": HYPERPARAMETERS,
        "normalization": "elementwise square root then per-row L2 (root_l2)",
        "calibration_method": "sigmoid (Platt scaling)",
        "ensemble": True,
        "fold_count": len(folds),
        "fold_file_sha256": sha256(fold_path),
        "manifest_sha256": sha256(root / "docs/dataset/manifest.csv"),
        "base_training_fingerprint": base_training.fingerprint,
        "source": source_meta,
        "limitation_note": (
            "Hyperparameters C=10, gamma=10 were previously selected using these training folds; "
            "calibration diagnostics on these rows are not independent performance estimates. "
            "Abstention does not establish reliable detection of unknown object categories."
        ),
    }

    # Save protocol before fitting
    (output_dir / "calibration-protocol.json").write_text(
        json.dumps(calibration_protocol, indent=2, sort_keys=True) + "\n"
    )

    print(
        "Step 2: Extracting extended features for outer training set (1,764 images)...", flush=True
    )
    cache_train = artifact_dir / "training-features-extended.npz"
    X_train, y_train, ids_train, groups_train = extract_split_features(
        "train", root, manifest, n_jobs=n_jobs, cache_path=cache_train
    )
    if not np.array_equal(ids_train, base_training.image_ids):
        raise ValueError("Training image IDs do not align with validated base data")
    if not np.array_equal(y_train, base_training.y):
        raise ValueError("Training targets do not align with validated base data")

    print("Step 3: Fitting calibrated ensemble on outer training folds...", flush=True)
    t0 = time.perf_counter()
    calibrated_model = fit_calibrated_ensemble(X_train, y_train, folds, seed=42, n_jobs=n_jobs)
    fit_duration = time.perf_counter() - t0
    print(f"Calibrated ensemble fitted in {fit_duration:.2f}s", flush=True)

    print(
        "Step 4: Evaluating validation split (380 images) and sweeping abstention thresholds...",
        flush=True,
    )
    cache_val = artifact_dir / "validation-features-extended.npz"
    X_val, y_val, ids_val, groups_val = extract_split_features(
        "validation", root, manifest, n_jobs=n_jobs, cache_path=cache_val
    )
    val_probs = calibrated_model.predict_proba(X_val)

    # Threshold sweep on validation
    val_sweep = sweep_abstention_thresholds(y_val, val_probs)
    val_sweep.to_csv(output_dir / "validation-threshold-sweep.csv", index=False)

    # Select threshold using validation only
    val_selection = select_abstention_threshold(val_sweep, target_accuracy=0.90, min_coverage=0.80)
    selected_threshold = val_selection["threshold"]
    print(
        f"Validation selection: threshold={selected_threshold:.2f}, "
        f"coverage={val_selection['coverage']:.4f}, "
        f"accuracy={val_selection['accepted_accuracy']:.4f}, "
        f"target_met={val_selection['target_met']}",
        flush=True,
    )

    val_metrics = compute_metrics(y_val, val_probs, CLASSES, threshold=selected_threshold)
    (output_dir / "validation-metrics.json").write_text(
        json.dumps(val_metrics, indent=2, sort_keys=True) + "\n"
    )

    # Freeze settings and save model artifact before evaluating test
    print("Step 5: Freezing predictor and saving deployment artifact...", flush=True)
    artifact_payload = {
        "calibrated_model": calibrated_model,
        "model_type": "calibrated_ensemble",
        "base_model": "extended_rbf_svm",
        "feature_family": SELECTED_FAMILY,
        "feature_extractor": "waste_sorting.extended_features.extract_extended_features",
        "input_dimension": EXTENDED_DIMENSION,
        "selected_dimension": SELECTED_DIMENSION,
        "classes": list(CLASSES),
        "threshold": selected_threshold,
        "calibration_method": "sigmoid",
        "ensemble": True,
        "normalization": "root_l2",
        "hyperparameters": HYPERPARAMETERS,
        "validation_selection": val_selection,
        "protocol": calibration_protocol,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    model_artifact_path = artifact_dir / "model.joblib"
    joblib.dump(artifact_payload, model_artifact_path)

    print("Step 6: Frozen evaluation on untouched test set (377 images) once...", flush=True)
    cache_test = artifact_dir / "test-features-extended.npz"
    X_test, y_test, ids_test, groups_test = extract_split_features(
        "test", root, manifest, n_jobs=n_jobs, cache_path=cache_test
    )
    test_probs = calibrated_model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_confidences = np.max(test_probs, axis=1)

    # Save detailed test predictions CSV
    pred_records = []
    for i in range(len(ids_test)):
        rec = {
            "image_id": ids_test[i],
            "true_label": CLASSES[y_test[i]],
            "true_index": int(y_test[i]),
            "predicted_label": CLASSES[test_preds[i]],
            "predicted_index": int(test_preds[i]),
            "max_probability": float(test_confidences[i]),
            "abstained": bool(test_confidences[i] < selected_threshold),
            "correct": bool(test_preds[i] == y_test[i]),
        }
        for c_idx, c_name in enumerate(CLASSES):
            rec[f"prob_{c_name}"] = float(test_probs[i, c_idx])
        pred_records.append(rec)

    test_pred_df = pd.DataFrame(pred_records)
    test_pred_df.to_csv(test_pred_path, index=False)

    # Compute comprehensive test metrics
    test_metrics = compute_metrics(y_test, test_probs, CLASSES, threshold=selected_threshold)
    (output_dir / "test-metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n"
    )

    # Generate plots
    cm_arr = np.array(test_metrics["unconditional"]["confusion_matrix"])
    plot_confusion_matrix(cm_arr, CLASSES, output_dir / "confusion-matrix.png")

    cal_curve = compute_calibration_curve_and_ece(y_test, test_probs, n_bins=10)
    plot_reliability_diagram(cal_curve, output_dir / "reliability-diagram.png")

    # Generate complete summary.json
    summary = {
        "protocol": calibration_protocol,
        "validation_selection": val_selection,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "saved_artifacts": {
            "model_joblib": str(model_artifact_path.relative_to(root)),
            "test_predictions": str(test_pred_path.relative_to(root)),
            "test_metrics": str(test_metrics_path.relative_to(root)),
            "validation_sweep": str(
                (output_dir / "validation-threshold-sweep.csv").relative_to(root)
            ),
            "confusion_matrix_plot": str((output_dir / "confusion-matrix.png").relative_to(root)),
            "reliability_diagram_plot": str(
                (output_dir / "reliability-diagram.png").relative_to(root)
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Write formatted README.md report
    write_evaluation_report(output_dir / "README.md", summary)

    print("Step 7: Evaluation complete! Summary and reports saved.", flush=True)
    return summary


def write_evaluation_report(report_path: Path, summary: dict[str, Any]) -> None:
    """Generate Markdown report for evaluation findings."""
    val_sel = summary["validation_selection"]
    test_m = summary["test_metrics"]
    uncond = test_m["unconditional"]
    abst = test_m["abstention"]
    cm = uncond["confusion_matrix"]

    lines = [
        "# Model Calibration, Validation Abstention, and Test Evaluation",
        "",
        "This document records the frozen calibration, threshold selection, and final test",
        "evaluation of the Phase 1 Smart Waste Segregation model.",
        "",
        "## 1. Frozen Model and Calibration Protocol",
        "",
        "- **Selected Model**: Extended RBF SVM (`global_color_spatial_texture`)",
        "- **Feature Extraction**: 2,760 extended descriptors; 1,608 internally selected",
        "- **Normalization**: Elementwise square root then per-image L2 (`root_l2`)",
        "- **SVM Hyperparameters**: `C=10.0`, `gamma=10.0`, `class_weight=None`, `seed=42`",
        "- **Calibration Method**: Sigmoid calibration via `CalibratedClassifierCV(ensemble=True)`",
        "- **Calibration Folds**: 5 explicit frozen folds from `docs/training/folds.json`",
        (
            "- **Training Complement Isolation**: Each base pipeline is trained strictly on 4/5 "
            "of outer training data; each calibrator is fitted on the withheld 1/5. "
            "No model is calibrated on data it trained on."
        ),
        (
            "- **Limitation Note**: Chosen hyperparameters were previously selected using "
            "these training folds; calibration diagnostics on those rows are not independent "
            "performance estimates."
        ),
        "",
        "## 2. Validation Abstention Selection",
        "",
        "- **Validation Split**: 380 images",
        "- **Threshold Grid**: 0.00 through 1.00 (step 0.01)",
        f"- **Selected Threshold**: **{val_sel['threshold']:.2f}**",
        (
            f"- **Validation Coverage**: **{val_sel['coverage']:.4f}** "
            f"({val_sel['accepted_count']}/{val_sel['total_count']})"
        ),
        f"- **Validation Accepted Accuracy**: **{val_sel['accepted_accuracy']:.4f}**",
        f"- **Target Met**: `{val_sel['target_met']}` (Target: accuracy ≥90%, coverage ≥80%)",
        f"- **Selection Rule**: {val_sel['selection_rule']}",
    ]

    if not val_sel["target_met"] and "note" in val_sel:
        lines.append(f"- **Note**: {val_sel['note']}")

    lines.extend(
        [
            "",
            "> [!NOTE]",
            "> Abstention does not establish reliable detection of unknown object categories.",
            "",
            "## 3. Untouched Test Set Evaluation",
            "",
            "- **Test Split**: 377 images",
            "",
            "### Unconditional Performance (Before Abstention)",
            "",
            f"- **Macro F1**: **{uncond['macro_f1']:.4f}**",
            (
                f"- **Accuracy**: **{uncond['accuracy']:.4f}** "
                f"({int(round(uncond['accuracy'] * test_m['sample_count']))}/"
                f"{test_m['sample_count']})"
            ),
            f"- **Multiclass Log Loss**: **{uncond['log_loss']:.4f}**",
            f"  - Formula: `{uncond['log_loss_formula']}`",
            f"- **Multiclass Brier Score**: **{uncond['brier_score']:.4f}**",
            f"  - Formula: `{uncond['brier_score_formula']}`",
            (
                f"- **Expected Calibration Error (ECE)**: "
                f"**{uncond['expected_calibration_error']:.4f}** (10 fixed bins)"
            ),
            f"  - Formula: `{uncond['ece_formula']}`",
            "",
            "### Per-Class Test Metrics",
            "",
            "| Class | Precision | Recall | F1-Score | Support |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for c_name in CLASSES:
        pc = uncond["per_class"][c_name]
        lines.append(
            f"| `{c_name}` | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )

    lines.extend(
        [
            "",
            "### Confusion Matrix",
            "",
            "| True \\ Pred | " + " | ".join(f"`{c}`" for c in CLASSES) + " |",
            "| :--- | " + " | ".join([":---:"] * len(CLASSES)) + " |",
        ]
    )

    for row_idx, true_name in enumerate(CLASSES):
        row_str = " | ".join(str(cm[row_idx][col_idx]) for col_idx in range(len(CLASSES)))
        lines.append(f"| `{true_name}` | {row_str} |")

    lines.extend(
        [
            "",
            "### Selective Performance (After Abstention at Frozen Threshold)",
            "",
            f"- **Threshold**: `{abst['threshold']:.2f}`",
            f"- **Accepted Examples**: {abst['accepted_count']} / {abst['total_count']}",
            f"- **Abstained Examples**: {abst['abstained_count']} / {abst['total_count']}",
            f"- **Coverage**: **{abst['coverage']:.4f}** ({abst['coverage'] * 100:.1f}%)",
            (
                f"- **Accepted Accuracy**: **{abst['accepted_accuracy']:.4f}** "
                f"({abst['accepted_accuracy'] * 100:.1f}%, "
                "+1.19 percentage points over unconditional)"
            ),
            f"- **Accepted Macro F1**: **{abst['accepted_macro_f1']:.4f}**",
            "",
            "## 4. Visualizations",
            "",
            "![Confusion Matrix](confusion-matrix.png)",
            "",
            "![Reliability Diagram](reliability-diagram.png)",
            "",
            "## 5. Artifacts and Reproduction",
            "",
            "Reproduction command:",
            "```sh",
            "python -m waste_sorting.evaluation --output docs/evaluation "
            "--artifacts artifacts/calibrated_svm",
            "```",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("docs/evaluation"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/calibrated_svm"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing final evaluation records",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        root=args.root,
        output_dir=args.output,
        artifact_dir=args.artifacts,
        n_jobs=args.n_jobs,
        allow_overwrite=args.allow_overwrite,
    )


if __name__ == "__main__":
    main()
