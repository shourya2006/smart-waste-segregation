"""Export small, reviewable results/figures from a completed training-only study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.study / "study-summary.json").read_text())
    if summary["outer_holdout_predictions_generated"]:
        raise ValueError("Expected a training-only study")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "candidate-results.csv").write_bytes(
        (args.study / "candidate-results.csv").read_bytes()
    )
    rows = []
    for run in summary["runs"]:
        rows.append(
            {
                "phase": run["phase"],
                "model": run["model"],
                "feature_set": run["feature_set"],
                "normalization": run["normalization"],
                "dimensions": run["selected_feature_dimension"],
                "mean_cv_macro_f1": run["mean_cv_macro_f1"],
                "fold_std": run["std_cv_macro_f1"],
                "candidate_count": run["candidate_count"],
                "parameters": json.dumps(run["best_params"], sort_keys=True),
                "seconds": run["total_seconds"],
            }
        )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)

    diagnostics = summary["diagnostics"]
    labels = diagnostics["rbf_svm"]["class_order"]
    per_class = []
    for name, diagnostic in diagnostics.items():
        for label, values in diagnostic["per_class"].items():
            per_class.append(dict(model=name, label=label, **values))
    pd.DataFrame(per_class).to_csv(output / "per-class.csv", index=False)

    colors = ["#0072B2", "#D55E00", "#009E73"]
    model_names = {
        "linear_svm": "Best linear SVM",
        "rbf_svm": "Best RBF SVM",
        "random_forest_color": "Reference HSV Random Forest",
    }
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    width = 0.24
    for i, (name, diagnostic) in enumerate(diagnostics.items()):
        recall = [diagnostic["per_class"][label]["recall"] for label in labels]
        ax.bar(
            np.arange(len(labels)) + (i - 1) * width,
            recall,
            width=width,
            color=colors[i],
            label=model_names[name],
        )
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylim=(0, 1.05),
        ylabel="Recall",
        title="Per-class recall on training out-of-fold predictions",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(output / "per-class-recall.png", dpi=200)
    plt.close(fig)

    names = [name for name in ["random_forest_color", "rbf_svm"] if name in diagnostics]
    fig, axes = plt.subplots(
        1, len(names), figsize=(6 * len(names), 5.5), layout="constrained", squeeze=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        matrix = np.asarray(diagnostics[name]["confusion_matrix"])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(len(labels)):
            for col in range(len(labels)):
                ax.text(
                    col,
                    row,
                    f"{normalized[row, col]:.0%}\n({matrix[row, col]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[row, col] > 0.55 else "black",
                )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Predicted class",
            ylabel="True class",
            title=model_names[name],
        )
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=list(axes[0]), fraction=0.03, label="Fraction of true class")
    fig.suptitle("Training OOF confusion matrices — selection diagnostics", fontsize=12)
    fig.savefig(output / "confusion-matrices.png", dpi=200)
    plt.close(fig)

    lines = [
        "# Verified SVM follow-up results",
        "",
        "All scores use the fixed outer-training folds. Parameters and feature families "
        "were selected using these scores; they are not final-test estimates.",
        "",
        "| Model | Pooled OOF accuracy | Mean fold macro F1 | Pooled OOF macro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, diagnostic in diagnostics.items():
        lines.append(
            f"| {model_names[name]} | {diagnostic['accuracy']:.4f} | "
            f"{diagnostic['mean_fold_macro_f1']:.4f} | "
            f"{diagnostic['pooled_macro_f1']:.4f} |"
        )
    lines += [
        "",
        "![Training OOF recall](per-class-recall.png)",
        "",
        "![Training OOF confusions](confusion-matrices.png)",
        "",
        "See comparison.csv for all winning runs, candidate-results.csv for every "
        "candidate, per-class.csv for class metrics, and summary.json for full provenance.",
        "",
        "Outer validation and test predictions were not generated.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
