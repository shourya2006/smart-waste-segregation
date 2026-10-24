"""Generate report plots and markdown summary for Random Forest refinement study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.verify_rf_refinement import audit, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--update-docs", action="store_true")
    args = parser.parse_args()

    audit(args.root, args.study)

    study = json.loads((args.study / "summary.json").read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    for file in ["summary.json", "candidate-results.csv", "comparison.csv"]:
        (output / file).write_bytes((args.study / file).read_bytes())

    for file in ["per-class.csv", "confusion-matrix.csv", "training-oof-predictions.csv"]:
        (output / file).write_bytes((args.study / "diagnostics" / file).read_bytes())

    labels = study["diagnostics"]["class_order"]

    # Confusion matrix heatmap
    matrix = np.asarray(study["diagnostics"]["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Greens")
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
        title=f"Refined RF ({study['winner']['family']}): training OOF confusions",
    )
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Fraction of true class")
    fig.savefig(output / "confusion-matrix.png", dpi=200)
    plt.close(fig)

    # Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    per_class = study["diagnostics"]["per_class"]
    f1_scores = [per_class[k]["f1-score"] for k in labels]
    recalls = [per_class[k]["recall"] for k in labels]
    precisions = [per_class[k]["precision"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, precisions, width, label="Precision", color="#4A7BB0")
    ax.bar(x, recalls, width, label="Recall", color="#E69F00")
    ax.bar(x + width, f1_scores, width, label="F1-score", color="#009E73")
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.05),
        ylabel="Score",
        title=f"Per-class Metrics: Refined RF ({study['winner']['family']})",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(output / "per-class-metrics.png", dpi=200)
    plt.close(fig)

    write_report(args.root, args.study, output, update_docs=args.update_docs)
    print(f"Report generated in {output}")


if __name__ == "__main__":
    main()
