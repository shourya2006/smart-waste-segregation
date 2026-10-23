"""Publish compact tables/figures from the completed extended SVM study."""

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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads((args.study / "summary.json").read_text())
    reference = json.loads((args.reference / "study-summary.json").read_text())
    assert not study["outer_holdout_predictions_generated"]
    assert not reference["outer_holdout_predictions_generated"]
    assert (
        study["plan"]["baseline_training_fingerprint"] == reference["plan"]["training_fingerprint"]
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for file in ["summary.json", "candidate-results.csv"]:
        (output / file).write_bytes((args.study / file).read_bytes())
    for file in ["per-class.csv", "confusion-matrix.csv"]:
        (output / file).write_bytes((args.study / "diagnostics" / file).read_bytes())
    pd.DataFrame(study["runs"]).to_csv(output / "comparison.csv", index=False)
    models = {
        "HSV Random Forest": reference["diagnostics"]["random_forest_color"],
        "Compact HSV+LBP SVM": reference["diagnostics"]["rbf_svm"],
        "Extended colour+texture SVM": study["diagnostics"],
    }
    labels = study["diagnostics"]["class_order"]
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    colors = ["#7B8794", "#0072B2", "#009E73"]
    for i, ((name, diagnostic), color) in enumerate(zip(models.items(), colors, strict=True)):
        ax.bar(
            np.arange(6) + (i - 1) * 0.25,
            [diagnostic["per_class"][k]["f1-score"] for k in labels],
            width=0.25,
            color=color,
            label=name,
        )
    ax.set(
        xticks=range(6),
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Per-class F1",
        title="Training out-of-fold performance: selected models",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(output / "per-class-comparison.png", dpi=200)
    plt.close(fig)

    matrix = np.asarray(study["diagnostics"]["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for r in range(6):
        for c in range(6):
            ax.text(
                c,
                r,
                f"{normalized[r, c]:.0%}\n({matrix[r, c]})",
                ha="center",
                va="center",
                color="white" if normalized[r, c] > 0.55 else "black",
                fontsize=9,
            )
    ax.set(
        xticks=range(6),
        xticklabels=labels,
        yticks=range(6),
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title="Extended SVM: training OOF confusions",
    )
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Fraction of true class")
    fig.savefig(output / "confusion-matrix.png", dpi=200)
    plt.close(fig)
    lines = [
        "# Extended SVM results",
        "",
        "All scores use the original training folds and were used for model selection. "
        "They are exploratory; outer validation/test performance is unknown.",
        "",
        "| Model | Mean fold macro F1 | Pooled OOF accuracy |",
        "| --- | ---: | ---: |",
    ]
    for name, diag in models.items():
        lines.append(f"| {name} | {diag['mean_fold_macro_f1']:.4f} | {diag['accuracy']:.4f} |")
    winner = study["winner"]
    lines += [
        "",
        f"Selected family: `{winner['family']}`; {winner['dimension']} selected columns "
        f"from the full 2,760-dimensional input. Fold std: {winner['std_cv_macro_f1']:.4f}.",
        "",
        "Parameters: `" + json.dumps(winner["best_params"], sort_keys=True) + "`.",
        "",
        "![Class comparison](per-class-comparison.png)",
        "",
        "![Confusions](confusion-matrix.png)",
        "",
        "See comparison.csv for all four families, candidate-results.csv for all 72 "
        "candidates, per-class.csv for the winner, and summary.json for full provenance.",
        "",
        "[Protocol and reproduction](../extended-svm-study.md)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
