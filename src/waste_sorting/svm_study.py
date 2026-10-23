"""A declared two-stage SVM feature study using only the frozen training folds.

The same CV scores guide screening and refinement; they are exploratory selection
scores, not nested-CV or final-test estimates. No outer holdout is scored here.
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict

from waste_sorting.acquisition import sha256
from waste_sorting.models.svm import make_pipeline, parameter_grid
from waste_sorting.training import (
    _source_metadata,
    _validate_cv,
    feature_columns,
    load_or_create_folds,
    load_training_data,
    run_experiment,
)

FEATURE_SETS = {
    "color": ["color"],
    "hog": ["hog"],
    "lbp": ["lbp"],
    "color_hog": ["color", "hog"],
    "combined": ["color", "hog", "lbp"],
    "color_lbp": ["color", "lbp"],
}


def refinement_winners(screen_runs, top_k):
    """Rank RBF feature families only; ties use feature-set names deterministically."""
    runs = [r for r in screen_runs if r["model"] == "rbf_svm"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(runs):
        raise ValueError("refinement_top_k must be between one and the number of RBF feature sets")
    return sorted(runs, key=lambda r: (-r["mean_cv_macro_f1"], r["feature_set"]))[:top_k]


def refinement_grid(settings, dimension, normalization):
    """Gamma values are declared or dimension-based, never fitted to held-out rows."""
    if normalization == "standard":
        gamma = [
            "scale",
            *[v / dimension for v in settings["standard_gamma_dimension_multipliers"]],
        ]
    elif normalization == "root_l2":
        gamma = settings["root_l2_gamma"]
    else:
        raise ValueError("Unknown refinement normalization")
    config = {
        "training": {
            "rbf_svm": {
                "C": settings["refined_C"],
                "gamma": gamma,
                "class_weight": settings["class_weight"],
            }
        }
    }
    return parameter_grid("rbf_svm", config)


def training_diagnostics(data, folds, estimator, output, *, n_jobs=1):
    """Out-of-fold training predictions, with their selection optimism documented."""
    _validate_cv(data, folds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
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
            f1_score(
                data.y[valid], predicted[valid], labels=classes, average="macro", zero_division=0
            )
        )
        fold_accuracy.append(accuracy_score(data.y[valid], predicted[valid]))
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
        "accuracy": accuracy_score(data.y, predicted),
        "pooled_macro_f1": f1_score(
            data.y, predicted, labels=classes, average="macro", zero_division=0
        ),
        "fold_macro_f1": fold_f1,
        "mean_fold_macro_f1": float(np.mean(fold_f1)),
        "fold_accuracy": fold_accuracy,
        "per_class": {label: report[label] for label in data.classes},
        "confusion_matrix": matrix.tolist(),
        "outer_holdout_predictions_generated": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_study(root, cache, config_path, fold_path, output, *, n_jobs=1, resume=False):
    root, output, fold_path = Path(root).resolve(), Path(output), Path(fold_path)
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = config["svm_study"]
    names = settings["feature_sets"]
    if not names or len(names) != len(set(names)) or not set(names).issubset(FEATURE_SETS):
        raise ValueError("feature_sets must contain distinct known feature families")
    if config["training"]["selection_metric"] != "f1_macro":
        raise ValueError("The study selection metric must be f1_macro")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    if (
        isinstance(settings["linear_max_iter"], bool)
        or not isinstance(settings["linear_max_iter"], int)
        or settings["linear_max_iter"] < 1
    ):
        raise ValueError("linear_max_iter must be a positive integer")
    # Validate all grids and the refinement rule before fitting anything.
    grids = {name: parameter_grid(name, config) for name in ["linear_svm", "rbf_svm"]}
    for mode in ["standard", "root_l2"]:
        refinement_grid(settings, 1, mode)
    top_k = settings["refinement_top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(names):
        raise ValueError("Invalid refinement_top_k")
    data = load_training_data(cache, root, config_path)
    folds = load_or_create_folds(data, fold_path, config["training"]["cv_folds"], config["seed"])
    plan = {
        "schema_version": 1,
        "config": config,
        "training_fingerprint": data.fingerprint,
        "fold_file_sha256": sha256(fold_path),
        "source": _source_metadata(root),
        "screening_grids": grids,
        "n_jobs": n_jobs,
        "screening_models": ["linear_svm", "rbf_svm"],
        "refinement_rule": "Top K RBF feature families; ties broken by feature-set name",
        "refinement_normalizations": ["standard", "root_l2"],
        "diagnostic_models": ["best_linear_svm", "best_rbf_svm", "reference_hsv_random_forest"],
        "held_out_predictions_generated": False,
    }
    if resume:
        existing = json.loads((output / "study-plan.json").read_text())
        if existing != plan:
            raise ValueError("Cannot resume: data, folds, source, environment or settings changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "study-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    runs = []

    def execute(phase, model, family, mode, grid):
        folder = output / phase / f"{model}-{family}-{mode}"
        summary_path = folder / "summary.json"
        if resume and summary_path.is_file():
            result = json.loads(summary_path.read_text())
            if not (folder / "estimator.joblib").is_file():
                raise ValueError(f"Missing estimator for completed run {folder}")
        else:
            if folder.exists():
                folder.rename(folder.with_name(folder.name + f"-incomplete-{time.time_ns()}"))
            descriptors = FEATURE_SETS[family]
            print(f"{phase}: {model} / {family} / {mode}", flush=True)
            result = run_experiment(
                data,
                folds,
                make_pipeline(
                    model,
                    feature_columns(data, descriptors),
                    config["seed"],
                    mode,
                    settings["linear_max_iter"],
                ),
                grid,
                model_name=model,
                descriptors=descriptors,
                output_dir=folder,
                root=root,
                fold_path=fold_path,
                seed=config["seed"],
                n_jobs=n_jobs,
            )
            result.update(
                {
                    "phase": phase,
                    "feature_set": family,
                    "normalization": mode,
                    "linear_max_iter": settings["linear_max_iter"]
                    if model == "linear_svm"
                    else None,
                }
            )
            summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if (
            result["parameter_grid"] != grid
            or result["training_fingerprint"] != data.fingerprint
            or result["fold_file_sha256"] != plan["fold_file_sha256"]
        ):
            raise ValueError(f"Saved run does not match the declared study: {folder}")
        result = dict(result, artifact_directory=str(folder.relative_to(output)))
        runs.append(result)
        print(
            f"  macro F1: {result['mean_cv_macro_f1']:.4f} +/- {result['std_cv_macro_f1']:.4f}",
            flush=True,
        )
        return result

    screen = []
    for family in names:
        for model in ["linear_svm", "rbf_svm"]:
            screen.append(execute("screen", model, family, "standard", grids[model]))
    selected = refinement_winners(screen, top_k)
    for original in selected:
        family = original["feature_set"]
        dimension = len(feature_columns(data, FEATURE_SETS[family]))
        for mode in ["standard", "root_l2"]:
            execute("refine", "rbf_svm", family, mode, refinement_grid(settings, dimension, mode))

    diagnostics = {}
    for model in ["linear_svm", "rbf_svm"]:
        winner = sorted(
            [r for r in runs if r["model"] == model],
            key=lambda r: (
                -r["mean_cv_macro_f1"],
                r["feature_set"],
                r["normalization"],
                r["phase"],
            ),
        )[0]
        diagnostic_path = output / "diagnostics" / model
        if resume and (diagnostic_path / "summary.json").exists():
            diagnostic = json.loads((diagnostic_path / "summary.json").read_text())
        else:
            diagnostic = training_diagnostics(
                data,
                folds,
                joblib.load(output / winner["artifact_directory"] / "estimator.joblib"),
                diagnostic_path,
                n_jobs=n_jobs,
            )
        np.testing.assert_allclose(
            diagnostic["fold_macro_f1"], winner["fold_macro_f1"], rtol=0, atol=1e-12
        )
        diagnostics[model] = dict(diagnostic, selected_run=winner["artifact_directory"])

    # Reference uses only Daksh's declared HSV winner, never outer holdout labels.
    rf_path = root / "docs/training/rf-cv-summary.json"
    if rf_path.is_file():
        from waste_sorting.models.random_forest import make_pipeline as make_rf

        reference = next(
            r for r in json.loads(rf_path.read_text())["runs"] if r["descriptors"] == ["color"]
        )
        if reference["training_fingerprint"] != data.fingerprint or reference[
            "fold_file_sha256"
        ] != sha256(fold_path):
            raise ValueError("Reference RF used different data or folds")
        diagnostic_path = output / "diagnostics/random_forest_color"
        if resume and (diagnostic_path / "summary.json").exists():
            diagnostic = json.loads((diagnostic_path / "summary.json").read_text())
        else:
            estimator = make_rf(feature_columns(data, ["color"]), config["seed"], n_jobs=1)
            estimator.set_params(**reference["best_params"])
            diagnostic = training_diagnostics(
                data, folds, estimator, diagnostic_path, n_jobs=n_jobs
            )
        np.testing.assert_allclose(
            diagnostic["fold_macro_f1"], reference["fold_macro_f1"], rtol=0, atol=1e-12
        )
        diagnostics["random_forest_color"] = diagnostic

    tables = []
    for run in runs:
        table = pd.read_csv(output / run["artifact_directory"] / "cv_results.csv")
        for field in ["model", "phase", "feature_set", "normalization"]:
            table.insert(0, field, run[field])
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(output / "candidate-results.csv", index=False)
    result = {
        "schema_version": 1,
        "plan": plan,
        "runs": runs,
        "diagnostics": diagnostics,
        "refined_feature_sets": [r["feature_set"] for r in selected],
        "candidate_evaluations": sum(r["candidate_count"] for r in runs),
        "cv_fits": sum(r["candidate_count"] * r["n_splits"] for r in runs),
        "final_training_refits": len(runs),
        "diagnostic_cv_fits": len(diagnostics) * len(folds),
        "score_interpretation": "Exploratory training-CV selection; not nested CV or final test",
        "outer_holdout_predictions_generated": False,
    }
    (output / "study-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    parser.add_argument("--folds", type=Path, default=Path("docs/training/folds.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def relative(path):
        return path if path.is_absolute() else root / path

    result = run_study(
        root,
        relative(args.cache),
        relative(args.config),
        relative(args.folds),
        relative(args.output),
        n_jobs=args.n_jobs,
        resume=args.resume,
    )
    print(
        f"Completed {result['candidate_evaluations']} candidates and "
        f"{result['cv_fits']} CV fits; no outer holdout predictions generated."
    )


if __name__ == "__main__":
    main()
