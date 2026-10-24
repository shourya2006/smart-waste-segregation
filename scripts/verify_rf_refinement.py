"""Verify saved experiment records and generate comparison tables without fitting."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from waste_sorting.rf_refinement import (
    FAMILY_DEFINITIONS,
    PARAM_DISTRIBUTIONS,
    sample_configurations,
)

TOLERANCE = 1e-12
TIMING = (
    "Family CV search wall time measures all 24 candidates × 5 folds (120 fits), "
    "including parallel scheduling, fitting, prediction and scoring. It excludes feature "
    "extraction, the final winning estimator refit, serialization and OOF diagnostics. "
    "Final-refit time was not recorded. The preserved raw `seconds` field has this meaning."
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(
        a.shape == b.shape
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, atol=TOLERANCE, rtol=0),
        context,
    )


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_scores(run):
    scores = run["fold_macro_f1"]
    check(len(scores) == 5, "Expected five fold scores")
    close(np.mean(scores), run["mean_cv_macro_f1"], "Mean differs from fold scores")
    close(np.std(scores, ddof=0), run["std_cv_macro_f1"], "Std differs from fold scores")


def verify_diagnostic(diag, class_counts):
    matrix = np.asarray(diag["confusion_matrix"])
    labels = diag["class_order"]
    check(
        matrix.shape == (6, 6)
        and (matrix >= 0).all()
        and np.equal(matrix, matrix.astype(int)).all(),
        "Invalid confusion counts",
    )
    close(matrix.sum(axis=1), [class_counts[k] for k in labels], "Class supports differ")
    close(matrix.sum(), diag["training_samples"], "Diagnostic row count differs")
    accuracy = float(np.trace(matrix) / matrix.sum())
    close(accuracy, diag["accuracy"], "Historical accuracy differs from confusion counts")
    true = np.repeat(labels, matrix.sum(axis=1).astype(int))
    pred = np.concatenate([np.repeat(labels, row.astype(int)) for row in matrix])
    report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    close(report["macro avg"]["f1-score"], diag["pooled_macro_f1"], "Pooled F1 mismatch")
    for label in labels:
        for key in ["precision", "recall", "f1-score", "support"]:
            close(report[label][key], diag["per_class"][label][key], "Per-class mismatch")
    close(np.mean(diag["fold_macro_f1"]), diag["mean_fold_macro_f1"], "Diagnostic mean mismatch")
    check(diag["outer_holdout_predictions_generated"] is False, "Holdout scoring reported")
    return accuracy


def audit(root, study, artifacts=None):
    """Validate portable committed records; optionally cross-check local run artifacts."""
    root, study = Path(root).resolve(), Path(study).resolve()
    artifacts = Path(artifacts).resolve() if artifacts is not None else None
    training = root / "docs/training"
    summary = read_json(study / "summary.json")
    folds = read_json(training / "folds.json")
    plan = summary["plan"]
    fold_hash = digest(training / "folds.json")
    check(plan["fold_file_sha256"] == fold_hash, "Frozen fold hash mismatch")
    check(
        plan["baseline_training_fingerprint"] == folds["training_fingerprint"],
        "Baseline fingerprint mismatch",
    )
    check(plan["seed"] == 42 and folds["n_splits"] == 5, "Unexpected sampling/fold protocol")
    check(plan["param_distributions"] == PARAM_DISTRIBUTIONS, "Parameter space mismatch")
    configurations = sample_configurations(seed=plan["seed"], n_iter=24)
    check(plan["sampled_candidates"] == configurations, "Sampled configurations mismatch")
    check(
        len({json.dumps(p, sort_keys=True) for p in configurations}) == 24,
        "Duplicate sampled configurations",
    )
    families = [definition[0] for definition in FAMILY_DEFINITIONS]
    check(plan["families"] == families, "Family plan mismatch")
    check(
        (plan["candidate_count"], plan["cv_fits"], plan["final_refits"]) == (96, 480, 4),
        "Recorded fit budget mismatch",
    )
    candidates = pd.read_csv(study / "candidate-results.csv", keep_default_na=False)
    check(
        len(candidates) == 96 and set(candidates.family) == set(families),
        "Expected exactly 96 candidate rows across four families",
    )
    check([r["family"] for r in summary["runs"]] == families, "Family summaries mismatch")
    errors = []
    selected_indices = {}
    for run, definition in zip(summary["runs"], FAMILY_DEFINITIONS, strict=True):
        family = run["family"]
        check(
            run["dimension"] == definition[3] and run["candidates"] == 24,
            "Family dimension/candidate count mismatch",
        )
        rows = candidates[candidates.family == family]
        check(sorted(rows.candidate_index) == list(range(24)), "Missing/duplicate candidate index")
        for _, row in rows.iterrows():
            params = ast.literal_eval(row["params"])
            check(params == configurations[row.candidate_index], "Candidate configuration mismatch")
            for key, value in params.items():
                recorded = row["param_" + key]
                if value is None:
                    check(recorded == "", "Expanded parameter column mismatch")
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    close(float(recorded), value, "Expanded parameter column mismatch")
                else:
                    check(str(recorded) == str(value), "Expanded parameter column mismatch")
            scores = np.array([row[f"split{i}_test_score"] for i in range(5)], dtype=float)
            check(
                np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
                "Invalid fold score",
            )
            mean, std = float(np.mean(scores)), float(np.std(scores, ddof=0))
            close(mean, row.mean_cv_macro_f1, "Candidate mean differs from five scores")
            close(std, row.std_cv_macro_f1, "Candidate std differs from five scores")
            errors.append(max(abs(mean - row.mean_cv_macro_f1), abs(std - row.std_cv_macro_f1)))
        best = rows.sort_values(
            ["mean_cv_macro_f1", "std_cv_macro_f1", "candidate_index"],
            ascending=[False, True, True],
        ).iloc[0]
        selected_indices[family] = int(best.candidate_index)
        check(run["best_params"] == configurations[best.candidate_index], "Family winner mismatch")
        close(
            run["fold_macro_f1"],
            [best[f"split{i}_test_score"] for i in range(5)],
            "Family winning fold scores mismatch",
        )
        verify_scores(run)
        if artifacts is not None:
            check(
                read_json(artifacts / family / "summary.json") == run,
                "Local family summary mismatch",
            )
            local = pd.read_csv(artifacts / family / "candidate-results.csv", keep_default_na=False)
            pd.testing.assert_frame_equal(
                local.drop(columns="rank_test_score")
                .sort_values("candidate_index")
                .reset_index(drop=True),
                rows.sort_values("candidate_index").reset_index(drop=True),
            )
            check(
                local.sort_values("rank_test_score").candidate_index.tolist()
                == rows.sort_values(
                    ["mean_cv_macro_f1", "std_cv_macro_f1", "candidate_index"],
                    ascending=[False, True, True],
                ).candidate_index.tolist(),
                "Local ranks mismatch",
            )
    winner = sorted(
        summary["runs"], key=lambda r: (-r["mean_cv_macro_f1"], r["std_cv_macro_f1"], r["family"])
    )[0]
    check(summary["winner"] == winner, "Overall winner mismatch")
    check(summary["outer_holdout_predictions_generated"] is False, "Holdout scoring reported")
    oof_path = study / "training-oof-predictions.csv"
    if not oof_path.exists():
        oof_path = study / "diagnostics/training-oof-predictions.csv"
    oof = pd.read_csv(oof_path)
    manifest_path = root / "docs/dataset/manifest.csv"
    check(digest(manifest_path) == folds["provenance"]["manifest_sha256"], "Manifest hash mismatch")
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest.included & (manifest.split == "train")].set_index("image_id")
    check(len(oof) == len(train) == len(folds["assignments"]) == 1764, "OOF row count mismatch")
    check(
        oof.image_id.is_unique
        and set(oof.image_id) == set(train.index) == set(folds["assignments"]),
        "OOF must cover each outer-training image exactly once",
    )
    check(
        oof.fold.tolist() == oof.image_id.map(folds["assignments"]).tolist(),
        "OOF frozen fold assignment mismatch",
    )
    for column, expected in [("true_label", "label"), ("duplicate_group", "duplicate_group")]:
        check(
            oof[column].tolist() == train.loc[oof.image_id, expected].tolist(),
            "OOF labels/groups do not match manifest",
        )
    check(oof.groupby("duplicate_group").fold.nunique().max() == 1, "Group crosses CV folds")
    included = manifest[manifest.included]
    check(
        included.groupby("duplicate_group").split.nunique().max() == 1, "Group crosses outer splits"
    )
    diag = summary["diagnostics"]
    labels = diag["class_order"]
    check(oof.predicted_label.isin(labels).all(), "Unknown predicted label")
    fold_f1, fold_accuracy, sizes = [], [], []
    for i in range(5):
        sub = oof[oof.fold == i]
        check(
            sub.true_label.value_counts().to_dict() == folds["validation_class_counts"][i],
            "Frozen fold class counts mismatch",
        )
        fold_f1.append(
            f1_score(
                sub.true_label, sub.predicted_label, labels=labels, average="macro", zero_division=0
            )
        )
        fold_accuracy.append(accuracy_score(sub.true_label, sub.predicted_label))
        sizes.append(len(sub))
    close(fold_f1, winner["fold_macro_f1"], "Winner scores differ from OOF predictions")
    close(fold_f1, diag["fold_macro_f1"], "Diagnostic folds differ from OOF predictions")
    close(fold_accuracy, diag["fold_accuracy"], "Fold accuracy differs from OOF predictions")
    close(
        confusion_matrix(oof.true_label, oof.predicted_label, labels=labels),
        diag["confusion_matrix"],
        "Winner confusion matrix differs from OOF",
    )
    diagnostic_dir = oof_path.parent
    saved_matrix = pd.read_csv(diagnostic_dir / "confusion-matrix.csv", index_col=0)
    check(
        saved_matrix.index.tolist() == labels and saved_matrix.columns.tolist() == labels,
        "Saved confusion CSV class order mismatch",
    )
    close(saved_matrix.to_numpy(), diag["confusion_matrix"], "Saved confusion CSV mismatch")
    saved_per_class = pd.read_csv(diagnostic_dir / "per-class.csv").set_index("label")
    check(saved_per_class.index.tolist() == labels, "Saved per-class CSV labels mismatch")
    for label in labels:
        for metric in ["precision", "recall", "f1-score", "support"]:
            close(
                saved_per_class.loc[label, metric],
                diag["per_class"][label][metric],
                "Saved per-class CSV mismatch",
            )
    class_counts = train.label.value_counts().to_dict()
    verify_diagnostic(diag, class_counts)
    check(
        diag["training_fingerprint"] == winner["training_fingerprint"],
        "Winner fingerprint mismatch",
    )
    historical = {}
    svm = read_json(training / "svm-study-results/summary.json")
    extended = read_json(training / "extended-svm-results/summary.json")
    for study_record in [svm, extended]:
        check(
            study_record["plan"]["fold_file_sha256"] == fold_hash, "Historical fold hash mismatch"
        )
        for run in study_record["runs"]:
            verify_scores(run)
    for name, record in svm["diagnostics"].items():
        accuracy = verify_diagnostic(record, class_counts)
        close(
            np.average(record["fold_accuracy"], weights=sizes),
            accuracy,
            "Historical fold accuracies disagree with pooled accuracy",
        )
        historical[name] = accuracy
    historical["extended_rbf_svm"] = verify_diagnostic(extended["diagnostics"], class_counts)
    close(
        extended["diagnostics"]["fold_macro_f1"],
        extended["winner"]["fold_macro_f1"],
        "Extended winner/diagnostic mismatch",
    )
    close(
        np.average(extended["diagnostics"]["fold_accuracy"], weights=sizes),
        historical["extended_rbf_svm"],
        "Extended accuracy mismatch",
    )
    for name in ["svm-cv-summary.json", "rf-cv-summary.json"]:
        for run in read_json(training / name)["runs"]:
            check(run["fold_file_sha256"] == fold_hash, "Historical baseline fold mismatch")
            check(
                run["training_fingerprint"] == folds["training_fingerprint"],
                "Historical baseline fingerprint mismatch",
            )
            verify_scores(run)
    if artifacts is not None:
        check(read_json(artifacts / "plan.json") == plan, "Local plan mismatch")
        for name in ["summary.json", "candidate-results.csv", "comparison.csv"]:
            check(
                digest(artifacts / name) == digest(study / name), "Local/exported record mismatch"
            )
        check(
            digest(artifacts / "diagnostics/training-oof-predictions.csv") == digest(oof_path),
            "Local/exported OOF mismatch",
        )
    sources = [
        study / "summary.json",
        study / "candidate-results.csv",
        oof_path,
        training / "folds.json",
        manifest_path,
        training / "svm-cv-summary.json",
        training / "rf-cv-summary.json",
        training / "svm-study-results/summary.json",
        training / "extended-svm-results/summary.json",
        root / "src/waste_sorting/rf_refinement.py",
    ]
    result = {
        "status": "passed",
        "tolerance_absolute": TOLERANCE,
        "std_ddof": 0,
        "candidates_verified": 96,
        "recorded_fold_scores_verified": 480,
        "identical_sampled_configurations_per_family": 24,
        "max_candidate_mean_std_absolute_error": max(errors),
        "family_winner_candidate_indices": selected_indices,
        "fold_file_sha256": fold_hash,
        "fold_sizes": sizes,
        "oof_rows": len(oof),
        "correct_predictions": int((oof.true_label == oof.predicted_label).sum()),
        "winner_fold_macro_f1": fold_f1,
        "winner_accuracy": diag["accuracy"],
        "winner_pooled_macro_f1": diag["pooled_macro_f1"],
        "historical_accuracy_from_confusion_counts": historical,
        "local_artifacts_cross_checked": artifacts is not None,
        "fitting_performed": False,
        "fold_evidence": "Recorded plan hash, shared runner folds, manifest identities/groups, "
        "and winner OOF assignments agree. Nonwinning candidate predictions/indices were not "
        "saved; their fold use is supported by the shared task construction, not a separate trace.",
        "timing_definition": TIMING,
        "source_sha256": {
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest(p)
            for p in sources
        },
    }
    return summary, result


def comparison_rows(root, summary):
    training = Path(root) / "docs/training"
    svm = read_json(training / "svm-study-results/summary.json")
    baseline = read_json(training / "svm-cv-summary.json")
    rf = read_json(training / "rf-cv-summary.json")["runs"][0]
    extended = read_json(training / "extended-svm-results/summary.json")
    rows = []

    def add(name, representation, run, source, diagnostic=None, accuracy_source=""):
        verify_scores(run)
        if diagnostic:
            close(
                diagnostic["fold_macro_f1"],
                run["fold_macro_f1"],
                "Comparison run/diagnostic mismatch",
            )
        rows.append(
            dict(
                model=name,
                representation=representation,
                selected_dimensions=run.get("dimension", run.get("selected_feature_dimension")),
                mean_cv_macro_f1=run["mean_cv_macro_f1"],
                std_cv_macro_f1=run["std_cv_macro_f1"],
                pooled_oof_accuracy=diagnostic["accuracy"] if diagnostic else None,
                metrics_source=source,
                accuracy_source=accuracy_source
                or "Not recorded: no saved OOF predictions or diagnostic summary",
            )
        )

    for name, run in zip(
        ["Dummy baseline", "Original linear SVM", "Original RBF SVM"], baseline["runs"], strict=True
    ):
        add(name, "HSV+HOG+LBP", run, f"svm-cv-summary.json → runs.{run['model']}")
    add(
        "Original HSV RF",
        "HSV",
        rf,
        "rf-cv-summary.json → runs[0]",
        svm["diagnostics"]["random_forest_color"],
        "svm-study-results/summary.json → diagnostics.random_forest_color.accuracy",
    )
    compact = max(
        (r for r in svm["runs"] if r["model"] == "rbf_svm"), key=lambda r: r["mean_cv_macro_f1"]
    )
    add(
        "Compact RBF SVM",
        "HSV+LBP",
        compact,
        "svm-study-results/summary.json → best RBF run",
        svm["diagnostics"]["rbf_svm"],
        "svm-study-results/summary.json → diagnostics.rbf_svm.accuracy",
    )
    add(
        "Refined RF",
        "Extended spatial color+texture",
        summary["winner"],
        "rf-refinement-results/summary.json → winner",
        summary["diagnostics"],
        "rf-refinement-results/training-oof-predictions.csv; summary.json → diagnostics.accuracy",
    )
    add(
        "Extended RBF SVM",
        "Extended global color+spatial texture",
        extended["winner"],
        "extended-svm-results/summary.json → winner",
        extended["diagnostics"],
        "extended-svm-results/summary.json → diagnostics.accuracy",
    )
    return rows


def markdown_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def generated_sections(root, summary, result):
    families = []
    for run in summary["runs"]:
        families.append(
            dict(
                family=run["family"],
                input_dimensions=1806 if run["source_type"] == "original" else 2760,
                selected_dimensions=run["dimension"],
                mean_cv_macro_f1=run["mean_cv_macro_f1"],
                std_cv_macro_f1=run["std_cv_macro_f1"],
                pooled_oof_accuracy=summary["diagnostics"]["accuracy"]
                if run["family"] == summary["winner"]["family"]
                else None,
                family_cv_search_wall_seconds=run["seconds"],
                best_params=json.dumps(run["best_params"], sort_keys=True),
            )
        )
    baseline_rows = []
    for filename in ["svm-cv-summary.json", "rf-cv-summary.json"]:
        for run in read_json(Path(root) / "docs/training" / filename)["runs"]:
            baseline_rows.append(
                [
                    run["model"],
                    "+".join(run["descriptors"]),
                    run["selected_feature_dimension"],
                    f"{run['mean_cv_macro_f1']:.6f}",
                    f"{run['std_cv_macro_f1']:.6f}",
                    f"`{json.dumps(run['best_params'], sort_keys=True)}`",
                    f"{run['total_seconds']:.1f}",
                ]
            )
    baseline_table = markdown_table(
        [
            "Model",
            "Descriptors",
            "Selected dims",
            "Mean CV macro F1",
            "Fold std",
            "Selected parameters",
            "Search + final refit wall time (s)",
        ],
        baseline_rows,
    ) + (
        "\n\nGenerated from `svm-cv-summary.json` and `rf-cv-summary.json`. "
        "These historical `total_seconds` measurements include GridSearchCV search and final "
        "refit; they differ from the refinement runner's CV-search-only `seconds` field."
    )
    family_table = markdown_table(
        [
            "Family",
            "Input dims",
            "Selected dims",
            "Mean CV macro F1",
            "Fold std",
            "Pooled OOF accuracy",
            "Family CV search wall time (s)",
            "Best parameters",
        ],
        [
            [
                f"`{r['family']}`",
                r["input_dimensions"],
                r["selected_dimensions"],
                f"{r['mean_cv_macro_f1']:.6f}",
                f"{r['std_cv_macro_f1']:.6f}",
                "Not recorded"
                if r["pooled_oof_accuracy"] is None
                else f"{r['pooled_oof_accuracy']:.6f}",
                f"{r['family_cv_search_wall_seconds']:.1f}",
                f"`{r['best_params']}`",
            ]
            for r in families
        ],
    )
    comparisons = comparison_rows(root, summary)
    cross_table = markdown_table(
        [
            "Model",
            "Representation",
            "Selected dims",
            "Mean CV macro F1",
            "Fold std",
            "Pooled OOF accuracy",
        ],
        [
            [
                r["model"],
                r["representation"],
                r["selected_dimensions"],
                f"{r['mean_cv_macro_f1']:.6f}",
                f"{r['std_cv_macro_f1']:.6f}",
                "Not recorded"
                if r["pooled_oof_accuracy"] is None
                else f"{r['pooled_oof_accuracy']:.6f}",
            ]
            for r in comparisons
        ],
    )
    validation = (
        f"All {result['candidates_verified']} candidates use the same 24 sampled configurations "
        "by index across four families. All 480 recorded fold scores reproduce their candidate "
        f"means and population standard deviations (ddof=0; tolerance {TOLERANCE:g}; "
        f"maximum error {result['max_candidate_mean_std_absolute_error']:.3g}). "
        "All family winners and the overall selection agree with the recorded tie-break rules.\n\n"
        f"Frozen fold SHA-256: `{result['fold_file_sha256']}`. Fold sizes: "
        f"`{result['fold_sizes']}`. The 1,764 unique OOF rows match the training manifest, labels, "
        "duplicate groups and every frozen fold assignment; "
        "groups do not cross folds or outer splits.\n\n"
        f"Winner fold macro F1 reproduced from predictions: `{result['winner_fold_macro_f1']}`. "
        f"Accuracy: **{result['winner_accuracy']}** ({result['correct_predictions']} / 1,764); "
        f"pooled macro F1: **{result['winner_pooled_macro_f1']}**. "
        "Confusion counts and per-class metrics also agree.\n\n"
        "Historical diagnostic accuracies were recomputed from integer confusion counts and "
        "checked against fold-size-weighted accuracies. The compact study's linear-SVM diagnostic "
        "belongs to its HSV+LBP model, not the original combined-feature linear baseline.\n\n"
        + result["fold_evidence"]
        + "\n\n"
        "The completed experiment was not refitted. These are exploratory selection-CV scores, "
        "not untouched-test estimates. See `validation.json` for exact values and source hashes."
    )
    return (
        families,
        comparisons,
        {
            "families": family_table + "\n\n" + TIMING,
            "baselines": baseline_table,
            "comparison": cross_table,
            "validation": validation,
        },
    )


def replace_section(text, name, content):
    start, end = f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"
    check(text.count(start) == text.count(end) == 1, f"Missing generated markers for {name}")
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + content + "\n" + end + after


def write_report(root, study, output, *, artifacts=None, update_docs=False):
    summary, result = audit(root, study, artifacts)
    families, comparisons, sections = generated_sections(root, summary, result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(families).to_csv(output / "family-comparison.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "cross-study-comparison.csv", index=False)
    (output / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    evidence = "\n".join(f"- **{r['model']}**: `{r['accuracy_source']}`." for r in comparisons)
    (output / "README.md").write_text(
        "# Random Forest refinement: verified results\n\n"
        "Generated from saved records; no experiment fitting. Do not hand-edit these tables.\n\n"
        "## Feature families\n\n" + sections["families"] + "\n\n"
        "Only the overall winner has saved OOF diagnostics. "
        "Other family accuracies are not recorded.\n\n"
        "## Cross-study comparison\n\n" + sections["comparison"] + "\n\n" + evidence + "\n\n"
        "Full-precision metrics and source paths are in `cross-study-comparison.csv`. "
        "`comparison.csv`, `candidate-results.csv` and `summary.json` "
        "preserve the raw experiment.\n\n"
        "## Verification\n\n" + sections["validation"] + "\n\n"
        "![Confusion matrix](confusion-matrix.png)\n\n![Per-class metrics](per-class-metrics.png)\n"
    )
    if update_docs:
        for path, names in [
            (
                Path(root) / "docs/training/rf-refinement-study.md",
                ["families", "comparison", "validation"],
            ),
            (Path(root) / "docs/member-notes/daksh-batra.md", ["families"]),
        ]:
            text = path.read_text()
            for name in names:
                text = replace_section(text, name, sections[name])
            path.write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifacts", type=Path, help="Optional original local experiment directory"
    )
    parser.add_argument("--update-docs", action="store_true")
    args = parser.parse_args()
    study = args.study or args.root / "docs/training/rf-refinement-results"
    result = write_report(
        args.root,
        study,
        args.output or study,
        artifacts=args.artifacts,
        update_docs=args.update_docs,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "candidates_verified",
                    "oof_rows",
                    "winner_accuracy",
                    "fitting_performed",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
