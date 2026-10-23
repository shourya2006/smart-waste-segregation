"""Validated training inputs, frozen group-aware CV, and shared experiment logging.

Validation/test rows are checked for manifest alignment and group isolation, then
discarded before returning any model inputs. No held-out predictions are made here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import tomllib
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from waste_sorting.acquisition import sha256
from waste_sorting.features import get_descriptor_boundaries, get_feature_names, load_features


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TrainingData:
    """Only outer-training rows, always in audited image-ID order."""

    X: np.ndarray
    y: np.ndarray
    image_ids: np.ndarray
    groups: np.ndarray
    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    boundaries: dict
    provenance: dict
    fingerprint: str


def load_training_data(cache_path, root, config_path=None):
    """Reject misaligned/tampered cache contents before selecting training rows."""
    root = Path(root).resolve()
    config_file = Path(config_path) if config_path else root / "configs/phase1.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    data = load_features(cache_path, root=root, expected_config=config)
    manifest = pd.read_csv(root / "docs/dataset/manifest.csv", keep_default_na=False)
    required = {"image_id", "label", "included", "split", "duplicate_group"}
    if not required.issubset(manifest.columns):
        raise ValueError("Manifest missing training identity/split columns")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs")
    if not manifest["included"].isin([True, False]).all():
        raise ValueError("Manifest inclusion flags must be boolean")
    included = manifest.loc[manifest["included"] == True].sort_values("image_id")  # noqa: E712
    if included.empty or not included["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Included images must have train/validation/test split assignments")
    if set(included["split"]) != {"train", "validation", "test"}:
        raise ValueError("All three audited outer splits must be present")
    if (included["duplicate_group"] == "").any():
        raise ValueError("Every included image needs a duplicate-group ID")
    if (included.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Duplicate group crosses outer split boundaries")

    classes = tuple(config["classes"])
    if (
        len(classes) != 6
        or len(set(classes)) != 6
        or not all(isinstance(c, str) and c for c in classes)
    ):
        raise ValueError("Expected six distinct, nonempty class names")
    if tuple(data["classes"]) != classes:
        raise ValueError("Cached class order does not match configured class order")
    for array, column in [
        ("image_ids", "image_id"),
        ("labels", "label"),
        ("splits", "split"),
        ("duplicate_groups", "duplicate_group"),
    ]:
        if not np.array_equal(data[array], included[column].to_numpy()):
            raise ValueError(f"Cached {array} differ from included manifest rows in image-ID order")
    if not included["label"].isin(classes).all():
        raise ValueError("Manifest contains unknown class labels")
    target_map = {label: i for i, label in enumerate(classes)}
    expected_y = included["label"].map(target_map).to_numpy(dtype=np.int64)
    if data["targets"].dtype.kind not in "iu" or not np.array_equal(data["targets"], expected_y):
        raise ValueError("Cached integer targets do not match manifest labels and class order")
    names = tuple(get_feature_names(config))
    boundaries = get_descriptor_boundaries(config)
    if tuple(data["feature_names"]) != names or data["boundaries"] != boundaries:
        raise ValueError("Cached feature names/boundaries do not match configuration")
    matrix = data["features"]
    if matrix.shape != (len(included), len(names)) or matrix.dtype != np.float32:
        raise ValueError("Cache must contain an aligned (images, features) float32 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    if (data["metadata"].get("sample_count"), data["metadata"].get("feature_dim")) != matrix.shape:
        raise ValueError("Cache matrix shape disagrees with metadata")

    mask = data["splits"] == "train"
    X = np.ascontiguousarray(matrix[mask])
    y = expected_y[mask]
    ids = data["image_ids"][mask].copy()
    groups = data["duplicate_groups"][mask].copy()
    if set(y) != set(range(len(classes))):
        raise ValueError("Training split must contain every configured class")
    provenance = {
        "manifest_sha256": data["metadata"]["manifest_sha256"],
        "feature_config_sha256": data["metadata"]["config_hash"],
        "feature_source_fingerprint": data["metadata"]["source_fingerprint"],
        "training_features_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "training_rows_sha256": _json_hash(
            list(zip(ids.tolist(), y.tolist(), groups.tolist(), strict=True))
        ),
        "classes": list(classes),
    }
    for array in (X, y, ids, groups):
        array.setflags(write=False)
    return TrainingData(
        X, y, ids, groups, classes, names, boundaries, provenance, _json_hash(provenance)
    )


def _validate_cv(data, folds):
    """Ensure every row is held out once and related images never cross a fold."""
    coverage = np.zeros(len(data.y), dtype=int)
    all_indices = np.arange(len(data.y))
    for train, valid in folds:
        for indices in (train, valid):
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or len(indices) == 0
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(data.y))
            ):
                raise ValueError("Invalid/duplicate/out-of-range fold indices")
        if not np.array_equal(np.sort(np.concatenate([train, valid])), all_indices):
            raise ValueError("Each fold must partition the complete outer training split")
        if set(data.groups[train]) & set(data.groups[valid]):
            raise ValueError("Duplicate group crosses a CV fold")
        for indices in (train, valid):
            if set(data.y[indices]) != set(range(len(data.classes))):
                raise ValueError("Each CV training/validation fold must contain all six classes")
        coverage[valid] += 1
    if len(folds) < 2 or not np.all(coverage == 1):
        raise ValueError("Every training image must be held out exactly once")


def make_folds(data, n_splits=5, seed=42):
    """Deterministic StratifiedGroupKFold; reject insufficient class/group support."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for label in range(len(data.classes)):
        if len(np.unique(data.groups[data.y == label])) < n_splits:
            raise ValueError("Each class needs at least n_splits distinct training groups")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(data.X, data.y, data.groups))
    _validate_cv(data, folds)
    return folds


def load_or_create_folds(data, path, n_splits=5, seed=42):
    """Freeze assignments by image ID; never silently replace an existing protocol.

    Both SVM and Random Forest owners call this interface on the complete training
    features before selecting descriptor subsets. Returned indices address data.X.
    """
    path = Path(path)
    folds = make_folds(data, n_splits, seed)
    assignment = {}
    for fold_id, (_, valid) in enumerate(folds):
        assignment.update({str(data.image_ids[i]): fold_id for i in valid})
    payload = {
        "schema_version": 1,
        "splitter": "StratifiedGroupKFold",
        "sklearn_version": sklearn.__version__,
        "n_splits": n_splits,
        "seed": seed,
        "training_fingerprint": data.fingerprint,
        "provenance": data.provenance,
        "assignments": assignment,
        "validation_class_counts": [
            {name: int(np.sum(data.y[valid] == i)) for i, name in enumerate(data.classes)}
            for _, valid in folds
        ],
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Saved folds differ from current training data/settings. "
                "Use the pinned environment and approved fold file; "
                "use a new path for a deliberate protocol change."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return folds


def feature_columns(data, descriptors=("color", "hog", "lbp")):
    """Return canonical columns; fit pipelines on full cached input vectors."""
    selected = set(descriptors)
    if not selected or not selected.issubset(data.boundaries):
        raise ValueError("Choose a nonempty subset of color, hog, lbp")
    return [
        i
        for name in ("color", "hog", "lbp")
        if name in selected
        for i in range(*data.boundaries[name])
    ]


def _source_metadata(root):
    hasher = hashlib.sha256()
    for path in sorted((root / "src/waste_sorting").rglob("*.py")):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "package_source_sha256": hasher.hexdigest(),
        "requirements_lock_sha256": sha256(root / "requirements.lock"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "scikit-image", "pandas", "joblib"]
        },
    }


def run_experiment(
    data,
    folds,
    estimator,
    param_grid,
    *,
    model_name,
    descriptors,
    output_dir,
    root,
    fold_path,
    seed=42,
    n_jobs=1,
):
    """Shared RF/SVM schema: training-CV search, then refit on outer training only.

    Scores here were used for selection and are not unbiased final-test estimates.
    The returned fitted estimator is uncalibrated. Confidence work belongs later.
    """
    _validate_cv(data, folds)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    root, fold_path = Path(root).resolve(), Path(fold_path)
    # Require the actual shared protocol file, not an unrelated file logged beside custom folds.
    persisted = json.loads(fold_path.read_text(encoding="utf-8"))
    expected_folds = load_or_create_folds(data, fold_path, len(folds), seed)
    if any(
        not np.array_equal(a, b)
        for pair_a, pair_b in zip(folds, expected_folds, strict=True)
        for a, b in zip(pair_a, pair_b, strict=True)
    ):
        raise ValueError("Experiment fold indices do not match the shared fold file")
    columns = feature_columns(data, descriptors)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scorer = make_scorer(
        f1_score, labels=list(range(len(data.classes))), average="macro", zero_division=0
    )
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=scorer,
        cv=folds,
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(data.X, data.y)
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {"model": model_name, "error_type": type(exc).__name__, "error": str(exc)}, indent=2
            )
            + "\n"
        )
        raise
    runtime = time.perf_counter() - started
    best = search.best_index_
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output / "cv_results.csv", index=False)
    fold_scores = [float(results.loc[best, f"split{i}_test_score"]) for i in range(len(folds))]
    summary = {
        "schema_version": 1,
        "model": model_name,
        "descriptors": [k for k in ("color", "hog", "lbp") if k in set(descriptors)],
        "selected_feature_dimension": len(columns),
        "input_feature_dimension": data.X.shape[1],
        "training_samples": len(data.y),
        "classes": list(data.classes),
        "seed": seed,
        "n_splits": len(folds),
        "selection_metric": "f1_macro",
        "mean_cv_macro_f1": float(search.best_score_),
        "std_cv_macro_f1": float(results.loc[best, "std_test_score"]),
        "fold_macro_f1": fold_scores,
        "best_params": search.best_params_,
        "parameter_grid": param_grid,
        "candidate_count": len(results),
        "total_seconds": runtime,
        "best_mean_fit_seconds": float(results.loc[best, "mean_fit_time"]),
        "best_mean_score_seconds": float(results.loc[best, "mean_score_time"]),
        "training_fingerprint": data.fingerprint,
        "provenance": data.provenance,
        "fold_file_sha256": sha256(fold_path),
        "splitter": persisted["splitter"],
        "source": _source_metadata(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fit_scope": "outer train only; selection by training CV; refit on all outer train",
        "calibrated": False,
        "held_out_predictions_generated": False,
        "score_interpretation": "Selection CV score; not an unbiased final-test estimate",
    }
    joblib.dump(search.best_estimator_, output / "estimator.joblib")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    """Create the shared folds or run Shourya's baselines on those folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--folds", type=Path, default=Path("docs/training/folds.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("folds", help="Create/verify shared training-only fold assignments")
    svm = sub.add_parser("svm", help="Search dummy/linear/RBF baselines on shared folds")
    svm.add_argument(
        "--models",
        nargs="+",
        choices=["dummy", "linear_svm", "rbf_svm"],
        default=["dummy", "linear_svm", "rbf_svm"],
    )
    svm.add_argument(
        "--descriptors", nargs="+", choices=["color", "hog", "lbp"], default=["color", "hog", "lbp"]
    )
    svm.add_argument("--output", type=Path, default=None)
    svm.add_argument("--n-jobs", type=int, default=1)

    rf_cmd = sub.add_parser(
        "rf", help="Search Random Forest pipelines and feature studies on shared folds"
    )
    rf_cmd.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["color", "hog", "lbp", "color_hog", "combined"],
        default=["color", "hog", "lbp", "color_hog", "combined"],
        help="Feature families to evaluate: color, hog, lbp, color_hog, combined",
    )
    rf_cmd.add_argument(
        "--descriptors",
        nargs="+",
        choices=["color", "hog", "lbp"],
        default=None,
        help="Explicit descriptor subset (overrides --feature-sets if provided)",
    )
    rf_cmd.add_argument("--output", type=Path, default=None)
    rf_cmd.add_argument("--n-jobs", type=int, default=1, help="Grid search worker count")
    rf_cmd.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=1,
        help="Estimator-level worker count for RandomForestClassifier",
    )

    args = parser.parse_args(argv)
    root = args.root.resolve()

    def relative(path):
        return path if path.is_absolute() else root / path

    config_file, cache_file, fold_file = map(relative, [args.config, args.cache, args.folds])
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    if config["training"]["selection_metric"] != "f1_macro":
        raise ValueError("The shared Phase 1 selection metric must be f1_macro")
    data = load_training_data(cache_file, root, config_file)
    folds = load_or_create_folds(data, fold_file, config["training"]["cv_folds"], config["seed"])
    print(f"Verified {len(data.y)} training rows and {len(folds)} shared folds: {fold_file}")
    if args.command == "folds":
        return

    if args.command == "svm":
        from waste_sorting.models.svm import make_pipeline, parameter_grid

        output = (
            relative(args.output)
            if args.output
            else root / "artifacts" / ("svm-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        )
        # Validate all requested grids before beginning any fitting.
        grids = {name: parameter_grid(name, config) for name in args.models}
        output.mkdir(parents=True, exist_ok=False)
        for model_name in dict.fromkeys(args.models):
            print(f"Training {model_name} on shared training folds...", flush=True)
            summary = run_experiment(
                data,
                folds,
                make_pipeline(model_name, feature_columns(data, args.descriptors), config["seed"]),
                grids[model_name],
                model_name=model_name,
                descriptors=args.descriptors,
                output_dir=output / model_name,
                root=root,
                fold_path=fold_file,
                seed=config["seed"],
                n_jobs=args.n_jobs,
            )
            print(
                f"{model_name}: CV macro F1 {summary['mean_cv_macro_f1']:.4f} "
                f"+/- {summary['std_cv_macro_f1']:.4f}",
                flush=True,
            )
        print(f"Training-only results written to {output}")
        return

    if args.command == "rf":
        from waste_sorting.models.random_forest import make_pipeline, parameter_grid

        output = (
            relative(args.output)
            if args.output
            else root / "artifacts" / ("rf-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        )
        grid = parameter_grid(config)

        feature_set_map = {
            "color": ["color"],
            "hog": ["hog"],
            "lbp": ["lbp"],
            "color_hog": ["color", "hog"],
            "combined": ["color", "hog", "lbp"],
        }
        if args.descriptors is not None:
            feature_runs = {"custom": args.descriptors}
        else:
            feature_runs = {name: feature_set_map[name] for name in args.feature_sets}

        output.mkdir(parents=True, exist_ok=False)
        for set_name, descriptors in feature_runs.items():
            run_name = f"random_forest_{set_name}"
            print(
                f"Training Random Forest on {set_name} ({descriptors}) "
                "on shared training folds...",
                flush=True,
            )
            summary = run_experiment(
                data,
                folds,
                make_pipeline(
                    feature_columns(data, descriptors),
                    seed=config["seed"],
                    n_jobs=args.estimator_n_jobs,
                ),
                grid,
                model_name="random_forest",
                descriptors=descriptors,
                output_dir=output / run_name,
                root=root,
                fold_path=fold_file,
                seed=config["seed"],
                n_jobs=args.n_jobs,
            )
            print(
                f"{run_name}: CV macro F1 {summary['mean_cv_macro_f1']:.4f} "
                f"+/- {summary['std_cv_macro_f1']:.4f} "
                f"(best: {summary['best_params']})",
                flush=True,
            )
        print(f"Training-only results written to {output}")
        return


if __name__ == "__main__":
    main()
