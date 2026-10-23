"""Bounded extended-histogram SVM study on the original training IDs/folds only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from waste_sorting.acquisition import sha256
from waste_sorting.extended_features import (
    DIMENSION,
    SETTINGS,
    extended_columns,
    extract_extended_features,
)
from waste_sorting.models.svm import make_pipeline
from waste_sorting.svm_study import training_diagnostics
from waste_sorting.training import (
    _json_hash,
    _source_metadata,
    _validate_cv,
    load_or_create_folds,
    load_training_data,
)

FAMILIES = [
    "global_color",
    "global_color_texture",
    "global_color_spatial_texture",
    "spatial_color_texture",
]
GRID = {
    "classifier__C": [10.0, 30.0, 100.0],
    "classifier__gamma": [1.0, 5.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}


def extract_training_matrix(base, root, *, n_jobs=1):
    """Read only training images, checking their bytes against the audited manifest."""
    manifest = pd.read_csv(root / "docs/dataset/manifest.csv", keep_default_na=False)
    manifest = manifest.set_index("image_id")

    def extract(image_id):
        row = manifest.loc[image_id]
        if row["split"] != "train" or not row["included"]:
            raise ValueError("Extended extraction accepts included training rows only")
        path = root / "data/raw/trashnet" / image_id
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {image_id}")
        with Image.open(path) as image:
            return extract_extended_features(image)

    matrix = np.asarray(
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(extract)(str(image_id)) for image_id in base.image_ids
        ),
        dtype=np.float32,
    )
    if matrix.shape != (len(base.y), DIMENSION) or not np.isfinite(matrix).all():
        raise ValueError("Invalid extended training matrix")
    return matrix


def derived_data(base, matrix):
    """Preserve identities, labels and groups while giving new features new provenance."""
    if matrix.shape != (len(base.y), DIMENSION) or matrix.dtype != np.float32:
        raise ValueError("Extended matrix shape/dtype mismatch")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Extended histograms must be finite and nonnegative")
    provenance = dict(base.provenance)
    provenance.update(
        {
            "baseline_training_fingerprint": base.fingerprint,
            "extended_feature_settings": SETTINGS,
            "extended_extractor_sha256": sha256(Path(__file__).with_name("extended_features.py")),
            "training_features_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        }
    )
    matrix.setflags(write=False)
    return replace(
        base,
        X=matrix,
        feature_names=tuple(f"extended_{i}" for i in range(DIMENSION)),
        boundaries={"extended": (0, DIMENSION)},
        provenance=provenance,
        fingerprint=_json_hash(provenance),
    )


def run(root, output, *, n_jobs=3):
    root, output = Path(root).resolve(), Path(output)
    base = load_training_data(root / "data/processed/features.npz", root)
    fold_path = root / "docs/training/folds.json"
    # Verify frozen folds using the original cache, then reuse EXACT indices for new features.
    folds = load_or_create_folds(base, fold_path)
    source = _source_metadata(root)
    plan = {
        "scope": "Exploratory selection CV; no outer validation/test predictions",
        "baseline_training_fingerprint": base.fingerprint,
        "fold_file_sha256": sha256(fold_path),
        "source": source,
        "settings": SETTINGS,
        "families": FAMILIES,
        "grid": GRID,
        "normalization": "elementwise square root then per-row L2",
        "n_jobs": n_jobs,
        "candidate_count": len(FAMILIES) * 18,
        "cv_fits": len(FAMILIES) * 18 * len(folds),
        "adaptation": "Follow-up declared after original SVM study reached 0.8042",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print("Extracting extended descriptors from audited training images only...", flush=True)
    started = time.perf_counter()
    matrix = extract_training_matrix(base, root, n_jobs=n_jobs)
    data = derived_data(base, matrix)
    _validate_cv(data, folds)
    extraction_seconds = time.perf_counter() - started
    np.savez_compressed(
        output / "training-features.npz",
        features=matrix,
        image_ids=base.image_ids,
        metadata=np.array(json.dumps(data.provenance)),
    )
    print(f"Extracted {matrix.shape} in {extraction_seconds:.1f}s", flush=True)
    scorer = make_scorer(f1_score, average="macro", labels=list(range(6)), zero_division=0)
    summaries, tables = [], []
    for family in FAMILIES:
        folder = output / family
        folder.mkdir()
        print(f"Searching {family}...", flush=True)
        estimator = make_pipeline("rbf_svm", extended_columns(family), normalization="root_l2")
        search = GridSearchCV(
            estimator,
            GRID,
            scoring=scorer,
            cv=folds,
            n_jobs=n_jobs,
            error_score="raise",
            refit=True,
        )
        started = time.perf_counter()
        search.fit(data.X, data.y)
        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "family", family)
        table.to_csv(folder / "candidate-results.csv", index=False)
        tables.append(table)
        summary = {
            "family": family,
            "mean_cv_macro_f1": float(search.best_score_),
            "std_cv_macro_f1": float(table.loc[search.best_index_, "std_test_score"]),
            "fold_macro_f1": [
                float(table.loc[search.best_index_, f"split{i}_test_score"])
                for i in range(len(folds))
            ],
            "best_params": search.best_params_,
            "candidates": len(table),
            "dimension": len(extended_columns(family)),
            "seconds": time.perf_counter() - started,
        }
        joblib.dump(search.best_estimator_, folder / "estimator.joblib")
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"  macro F1: {search.best_score_:.4f}", flush=True)
    winner = sorted(summaries, key=lambda s: (-s["mean_cv_macro_f1"], s["family"]))[0]
    diagnostic = training_diagnostics(
        data,
        folds,
        joblib.load(output / winner["family"] / "estimator.joblib"),
        output / "diagnostics",
        n_jobs=n_jobs,
    )
    np.testing.assert_allclose(
        diagnostic["fold_macro_f1"], winner["fold_macro_f1"], atol=1e-12, rtol=0
    )
    if _source_metadata(root) != source:
        raise ValueError("Source or environment changed during the extended study")
    pd.concat(tables, ignore_index=True).to_csv(output / "candidate-results.csv", index=False)
    result = {
        "plan": plan,
        "runs": summaries,
        "winner": winner,
        "diagnostics": diagnostic,
        "provenance": data.provenance,
        "training_fingerprint": data.fingerprint,
        "extraction_seconds": extraction_seconds,
        "outer_holdout_predictions_generated": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=3)
    args = parser.parse_args()
    run(args.root, args.output, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
