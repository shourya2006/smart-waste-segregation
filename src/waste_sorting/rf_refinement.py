"""Bounded Random Forest refinement study across four feature families.

Evaluates 24 predeclared parameter configurations sampled with seed 42 across:
1. original_hsv (32 features)
2. extended_global_color_texture (920 features)
3. extended_global_color_spatial_texture (1,608 features)
4. extended_spatial_color_texture (2,760 features)

Uses exact frozen folds from docs/training/folds.json and training-only data.
"""

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
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterSampler, cross_val_predict

from waste_sorting.acquisition import sha256
from waste_sorting.extended_features import (
    DIMENSION as EXTENDED_DIMENSION,
)
from waste_sorting.extended_features import (
    SETTINGS as EXTENDED_SETTINGS,
)
from waste_sorting.extended_features import (
    extended_columns,
)
from waste_sorting.extended_study import extract_training_matrix
from waste_sorting.models.random_forest import make_pipeline
from waste_sorting.training import (
    _json_hash,
    _source_metadata,
    _validate_cv,
    feature_columns,
    load_or_create_folds,
    load_training_data,
)

FAMILY_DEFINITIONS = [
    ("original_hsv", "original", ["color"], 32),
    ("extended_global_color_texture", "extended", "global_color_texture", 920),
    ("extended_global_color_spatial_texture", "extended", "global_color_spatial_texture", 1608),
    ("extended_spatial_color_texture", "extended", "spatial_color_texture", 2760),
]

PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [300, 600],
    "classifier__max_depth": [None, 20, 40],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", 0.25, 0.5, 1.0],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    "classifier__min_samples_split": [2],
    "classifier__bootstrap": [True],
}


def sample_configurations(seed: int = 42, n_iter: int = 24) -> list[dict]:
    """Deterministically sample candidate configurations."""
    sampler = ParameterSampler(PARAM_DISTRIBUTIONS, n_iter=n_iter, random_state=seed)
    return list(sampler)


def derived_extended_data(base, matrix):
    """Wrap extended matrix in TrainingData preserving identities and groups."""
    if matrix.shape != (len(base.y), EXTENDED_DIMENSION) or matrix.dtype != np.float32:
        raise ValueError("Extended matrix shape/dtype mismatch")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Extended histograms must be finite and nonnegative")
    provenance = dict(base.provenance)
    provenance.update(
        {
            "baseline_training_fingerprint": base.fingerprint,
            "extended_feature_settings": EXTENDED_SETTINGS,
            "extended_extractor_sha256": sha256(Path(__file__).with_name("extended_features.py")),
            "training_features_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        }
    )
    matrix.setflags(write=False)
    return replace(
        base,
        X=matrix,
        feature_names=tuple(f"extended_{i}" for i in range(EXTENDED_DIMENSION)),
        boundaries={"extended": (0, EXTENDED_DIMENSION)},
        provenance=provenance,
        fingerprint=_json_hash(provenance),
    )


def compute_diagnostics(data, folds, estimator, output_dir, *, n_jobs=1):
    """Compute out-of-fold training predictions and metrics for the winner."""
    _validate_cv(data, folds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predicted = cross_val_predict(
        estimator, data.X, data.y, cv=folds, n_jobs=n_jobs, method="predict"
    )
    classes = list(range(len(data.classes)))
    matrix = confusion_matrix(data.y, predicted, labels=classes)
    report = classification_report(
        data.y,
        predicted,
        labels=classes,
        target_names=list(data.classes),
        zero_division=0,
        output_dict=True,
    )
    fold_ids = np.empty(len(data.y), dtype=int)
    fold_f1, fold_accuracy = [], []
    for i, (_, valid) in enumerate(folds):
        fold_ids[valid] = i
        fold_f1.append(
            float(
                f1_score(
                    data.y[valid],
                    predicted[valid],
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            )
        )
        fold_accuracy.append(float(accuracy_score(data.y[valid], predicted[valid])))

    pd.DataFrame(
        {
            "image_id": data.image_ids,
            "duplicate_group": data.groups,
            "fold": fold_ids,
            "true_label": [data.classes[i] for i in data.y],
            "predicted_label": [data.classes[i] for i in predicted],
        }
    ).to_csv(output / "training-oof-predictions.csv", index=False)
    pd.DataFrame(matrix, index=data.classes, columns=data.classes).to_csv(
        output / "confusion-matrix.csv", index_label="true_label"
    )
    pd.DataFrame([dict(label=label, **report[label]) for label in data.classes]).to_csv(
        output / "per-class.csv", index=False
    )
    summary = {
        "scope": "outer-training OOF; parameters selected using these folds; exploratory",
        "training_samples": len(data.y),
        "training_fingerprint": data.fingerprint,
        "class_order": list(data.classes),
        "accuracy": float(accuracy_score(data.y, predicted)),
        "pooled_macro_f1": float(
            f1_score(data.y, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "fold_macro_f1": fold_f1,
        "mean_fold_macro_f1": float(np.mean(fold_f1)),
        "fold_accuracy": fold_accuracy,
        "per_class": {label: report[label] for label in data.classes},
        "confusion_matrix": matrix.tolist(),
        "outer_holdout_predictions_generated": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_rf_refinement(root, output, *, n_jobs=4, seed=42):
    root = Path(root).resolve()
    output = Path(output).resolve()
    base = load_training_data(root / "data/processed/features.npz", root)
    fold_path = root / "docs/training/folds.json"
    folds = load_or_create_folds(base, fold_path)

    source = _source_metadata(root)
    sampled_candidates = sample_configurations(seed=seed, n_iter=24)

    plan = {
        "scope": "Exploratory RF refinement selection CV; no outer validation/test predictions",
        "seed": seed,
        "baseline_training_fingerprint": base.fingerprint,
        "fold_file_sha256": sha256(fold_path),
        "source": source,
        "param_distributions": PARAM_DISTRIBUTIONS,
        "sampled_candidates": sampled_candidates,
        "candidate_count": len(FAMILY_DEFINITIONS) * len(sampled_candidates),
        "cv_fits": len(FAMILY_DEFINITIONS) * len(sampled_candidates) * len(folds),
        "final_refits": len(FAMILY_DEFINITIONS),
        "families": [f[0] for f in FAMILY_DEFINITIONS],
        "n_jobs": n_jobs,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print("Checking / extracting extended training matrix...", flush=True)
    started_ext = time.perf_counter()
    extended_matrix = extract_training_matrix(base, root, n_jobs=n_jobs)
    extended_data = derived_extended_data(base, extended_matrix)
    _validate_cv(extended_data, folds)
    extraction_seconds = time.perf_counter() - started_ext
    np.savez_compressed(
        output / "extended-training-features.npz",
        features=extended_matrix,
        image_ids=base.image_ids,
        metadata=np.array(json.dumps(extended_data.provenance)),
    )
    print(
        f"Extended matrix ready ({extended_matrix.shape}) in {extraction_seconds:.1f}s",
        flush=True,
    )

    family_summaries = []
    all_candidate_rows = []

    def evaluate_fold_candidate(estimator_template, params, train_idx, valid_idx, X, y):
        model = clone(estimator_template)
        model.set_params(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[valid_idx])
        score = f1_score(
            y[valid_idx], preds, average="macro", labels=list(range(6)), zero_division=0
        )
        return float(score)

    for family_name, source_type, spec, expected_dim in FAMILY_DEFINITIONS:
        family_dir = output / family_name
        family_dir.mkdir(parents=True, exist_ok=False)
        print(f"\n--- Evaluating Family: {family_name} (dim={expected_dim}) ---", flush=True)

        if source_type == "original":
            cur_data = base
            cols = feature_columns(base, spec)
        else:
            cur_data = extended_data
            cols = extended_columns(spec)

        if len(cols) != expected_dim:
            raise ValueError(
                f"Column count mismatch for {family_name}: got {len(cols)}, expected {expected_dim}"
            )

        pipe_template = make_pipeline(cols, seed=seed)

        tasks = []
        for cand_idx, params in enumerate(sampled_candidates):
            for fold_idx, (train_idx, valid_idx) in enumerate(folds):
                tasks.append((cand_idx, fold_idx, params, train_idx, valid_idx))

        started_fam = time.perf_counter()
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(evaluate_fold_candidate)(
                pipe_template, params, train_idx, valid_idx, cur_data.X, cur_data.y
            )
            for cand_idx, fold_idx, params, train_idx, valid_idx in tasks
        )
        fam_seconds = time.perf_counter() - started_fam

        # Organize by candidate
        candidate_rows = []
        for cand_idx, params in enumerate(sampled_candidates):
            cand_scores = results[cand_idx * len(folds) : (cand_idx + 1) * len(folds)]
            mean_score = float(np.mean(cand_scores))
            std_score = float(np.std(cand_scores))

            row = {
                "family": family_name,
                "candidate_index": cand_idx,
                "mean_cv_macro_f1": mean_score,
                "std_cv_macro_f1": std_score,
                "split0_test_score": cand_scores[0],
                "split1_test_score": cand_scores[1],
                "split2_test_score": cand_scores[2],
                "split3_test_score": cand_scores[3],
                "split4_test_score": cand_scores[4],
                "param_classifier__n_estimators": params["classifier__n_estimators"],
                "param_classifier__max_depth": params["classifier__max_depth"],
                "param_classifier__min_samples_leaf": params["classifier__min_samples_leaf"],
                "param_classifier__max_features": params["classifier__max_features"],
                "param_classifier__class_weight": params["classifier__class_weight"],
                "param_classifier__min_samples_split": params["classifier__min_samples_split"],
                "param_classifier__bootstrap": params["classifier__bootstrap"],
                "params": str(params),
            }
            candidate_rows.append(row)
            all_candidate_rows.append(row)

        cand_df = pd.DataFrame(candidate_rows)
        # Deterministic sorting: highest mean_cv_macro_f1, lowest std, lowest candidate_index
        cand_df = cand_df.sort_values(
            by=["mean_cv_macro_f1", "std_cv_macro_f1", "candidate_index"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        cand_df["rank_test_score"] = range(1, len(cand_df) + 1)
        cand_df.to_csv(family_dir / "candidate-results.csv", index=False)

        best_row = cand_df.iloc[0]
        best_cand_idx = int(best_row["candidate_index"])
        best_params = sampled_candidates[best_cand_idx]

        print(
            f"  Winner: candidate {best_cand_idx} with macro F1: "
            f"{best_row['mean_cv_macro_f1']:.4f} +/- {best_row['std_cv_macro_f1']:.4f} "
            f"(params: {best_params})",
            flush=True,
        )

        # Refit winning pipeline on all outer training rows
        best_estimator = make_pipeline(cols, seed=seed)
        best_estimator.set_params(**best_params)
        best_estimator.fit(cur_data.X, cur_data.y)
        joblib.dump(best_estimator, family_dir / "estimator.joblib")

        fam_summary = {
            "family": family_name,
            "source_type": source_type,
            "dimension": expected_dim,
            "mean_cv_macro_f1": float(best_row["mean_cv_macro_f1"]),
            "std_cv_macro_f1": float(best_row["std_cv_macro_f1"]),
            "fold_macro_f1": [float(best_row[f"split{i}_test_score"]) for i in range(len(folds))],
            "best_params": best_params,
            "candidates": len(sampled_candidates),
            "seconds": fam_seconds,
            "training_fingerprint": cur_data.fingerprint,
        }
        (family_dir / "summary.json").write_text(
            json.dumps(fam_summary, indent=2, sort_keys=True) + "\n"
        )
        family_summaries.append(fam_summary)

    # Determine overall winner
    overall_winner = sorted(
        family_summaries,
        key=lambda s: (-s["mean_cv_macro_f1"], s["std_cv_macro_f1"], s["family"]),
    )[0]
    print("\n==========================================")
    print(
        f"OVERALL WINNER: {overall_winner['family']} with "
        f"mean CV macro F1 = {overall_winner['mean_cv_macro_f1']:.4f}"
    )
    print("==========================================\n", flush=True)

    winner_name = overall_winner["family"]
    winner_source_type = overall_winner["source_type"]
    winner_data = base if winner_source_type == "original" else extended_data
    winner_estimator = joblib.load(output / winner_name / "estimator.joblib")

    # Run diagnostics on overall winner
    diagnostics = compute_diagnostics(
        winner_data,
        folds,
        winner_estimator,
        output / "diagnostics",
        n_jobs=n_jobs,
    )

    np.testing.assert_allclose(
        diagnostics["fold_macro_f1"], overall_winner["fold_macro_f1"], atol=1e-12, rtol=0
    )

    pd.DataFrame(all_candidate_rows).to_csv(output / "candidate-results.csv", index=False)
    pd.DataFrame(family_summaries).to_csv(output / "comparison.csv", index=False)

    overall_result = {
        "plan": plan,
        "runs": family_summaries,
        "winner": overall_winner,
        "diagnostics": diagnostics,
        "extraction_seconds": extraction_seconds,
        "outer_holdout_predictions_generated": False,
    }
    (output / "summary.json").write_text(
        json.dumps(overall_result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Completed Random Forest refinement. Results written to {output}")
    return overall_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_rf_refinement(args.root, args.output, n_jobs=args.n_jobs, seed=args.seed)


if __name__ == "__main__":
    main()
