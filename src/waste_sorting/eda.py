"""Train-only EDA and local contact sheets for duplicate review."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from waste_sorting.acquisition import CLASSES


def review_sheets(root: Path, candidates: pd.DataFrame) -> list[Path]:
    """Render every candidate pair for explicit review before splitting."""
    destination = root / "artifacts/duplicate-review"
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(candidates), 12):
        batch = candidates.iloc[start : start + 12]
        sheet = Image.new("RGB", (1040, 6 * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for position, row in enumerate(batch.itertuples()):
            x, y = (position % 2) * 520, (position // 2) * 190
            title = f"Pair {start + position}: d={row.phash_distance}"
            draw.text((x + 6, y + 3), title, fill="black")
            for offset, field in enumerate(("left_id", "right_id")):
                identifier = getattr(row, field)
                with Image.open(root / "data/raw/trashnet" / identifier) as source:
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    thumb = ImageOps.contain(rgb, (250, 140))
                sheet.paste(thumb, (x + offset * 260 + 5, y + 20))
                draw.text((x + offset * 260 + 5, y + 164), identifier, fill="black")
        path = destination / f"pairs-{start:04d}.jpg"
        sheet.save(path, quality=92)
        paths.append(path)
    return paths


def create_eda(root: Path) -> None:
    reports = root / "docs/dataset"
    manifest = pd.read_csv(reports / "manifest.csv", keep_default_na=False)
    audit = json.loads((reports / "audit_summary.json").read_text())
    audit_settings = json.loads((reports / "audit_settings.json").read_text())
    review = pd.read_csv(reports / "duplicate_reviews.csv")
    train = manifest[manifest.included & manifest.split.eq("train")].copy()
    if train.empty:
        raise ValueError("Create dataset splits before running EDA")
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    counts = train.label.value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(CLASSES, counts, color="#226A80")
    ax.bar_label(bars, padding=3)
    ax.set(title="TrashNet training class balance", ylabel="Images")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.savefig(figures / "train-class-balance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].boxplot(
        [train.loc[train.label.eq(c), "brightness"] for c in CLASSES], tick_labels=CLASSES
    )
    axes[0].set(title="Training brightness by class", ylabel="Mean grayscale intensity (0–255)")
    axes[0].tick_params(axis="x", labelrotation=30)
    for channel, color in zip("rgb", ("#b63939", "#31844c", "#315b9a"), strict=True):
        means = [train.loc[train.label.eq(c), f"mean_{channel}"].mean() for c in CLASSES]
        axes[1].plot(CLASSES, means, "o-", color=color, label=channel.upper())
    axes[1].set(title="Training average channel intensities", ylabel="Mean intensity (0–255)")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].legend()
    fig.savefig(figures / "train-color-brightness.png", dpi=150)
    plt.close(fig)

    # Source-image grids remain local rather than redistributing dataset photos.
    destination = root / "artifacts/eda"
    destination.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 260, len(CLASSES) * 220), "white")
    draw = ImageDraw.Draw(sheet)
    for row, label in enumerate(CLASSES):
        samples = train[train.label.eq(label)].sample(n=min(4, int(counts[label])), random_state=42)
        for col, sample in enumerate(samples.itertuples()):
            x, y = col * 260, row * 220
            with Image.open(root / "data/raw/trashnet" / sample.image_id) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (250, 180))
            sheet.paste(thumb, (x + 5, y + 5))
            draw.text((x + 5, y + 192), sample.image_id, fill="black")
    sheet.save(destination / "training-samples.jpg", quality=92)

    total = pd.crosstab(manifest[manifest.included].label, manifest[manifest.included].split)
    source_dimensions = ", ".join(sorted(audit["dimensions"]))
    source_modes = ", ".join(sorted(audit["color_modes"]))
    lines = [
        "# TrashNet dataset audit and exploratory analysis",
        "",
        "These results describe the pinned source archive and the committed manifest.",
        "Only training images were used for visual samples and exploratory intensity statistics.",
        "All-file checks before splitting were limited to integrity, labels, and duplicate review.",
        "",
        "## Integrity and duplicate findings",
        "",
        f"Audited {audit['total_files']} files: {audit['readable_files']} readable, "
        f"{audit['unreadable_files']} unreadable. Original class counts match upstream: "
        f"{audit['matches_upstream_counts']}. "
        f"Source color modes: {source_modes}; resolutions: {source_dimensions}.",
        "",
        f"Excluded {audit['excluded_label_conflicts']} images from "
        f"{audit['label_conflict_groups']} identical-content groups with conflicting labels. "
        "Neither member is assigned an assumed corrected label. The source files are unchanged.",
        "",
        "| Excluded image | Conflicting content group |",
        "| --- | --- |",
    ]
    for row in manifest[manifest.exclusion_reason.eq("conflicting_labels")].itertuples():
        lines.append(f"| {row.image_id} | {row.canonical_id} |")
    lines.extend(
        [
            "",
            f"Reviewed {len(review)} perceptual-hash candidates at Hamming distance "
            f"≤ {audit_settings['phash_distance']}: "
            f"{int(review.decision.eq('same_item').sum())} same-item pairs, "
            f"{int(review.decision.eq('different_items').sum())} different-item pairs, and "
            f"{int(review.decision.eq('uncertain').sum())} uncertain pairs. "
            "Same-item and uncertain "
            "pairs stay within a partition. These decisions are Codex visual assessments; "
            "a teammate should review the uncertain cases before locking the modeling protocol.",
            "",
            "## Split inventory",
            "",
            "| Class | Train | Validation | Test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in CLASSES:
        lines.append(
            f"| {label} | {total.loc[label, 'train']} | "
            f"{total.loc[label, 'validation']} | {total.loc[label, 'test']} |"
        )
    lines.extend(
        [
            "",
            "## Training observations",
            "",
            f"The training partition contains {len(train)} images. The largest class is "
            f"{counts.idxmax()} ({counts.max()}); "
            f"the smallest is {counts.idxmin()} ({counts.min()}). "
            f"The ratio is {counts.max() / counts.min():.2f}:1. Use macro F1 and per-class recall, "
            "and compare class weighting during model development.",
            "",
            "![Training class counts](figures/train-class-balance.png)",
            "",
            "| Class | Median brightness | Mean red | Mean green | Mean blue |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, group in train.groupby("label", sort=True):
        lines.append(
            f"| {label} | {group.brightness.median():.1f} | "
            f"{group.mean_r.mean():.1f} | {group.mean_g.mean():.1f} | "
            f"{group.mean_b.mean():.1f} |"
        )
    dimensions = sorted({f"{row.width} × {row.height}" for row in train.itertuples()})
    lines.extend(
        [
            "",
            "![Training color and brightness](figures/train-color-brightness.png)",
            "",
            "Training resolutions: "
            + ", ".join(dimensions)
            + ". Preserve aspect ratio when resizing.",
            "",
            "The source was photographed on a white posterboard. Full-image color and "
            "brightness measurements mix object and background signals. Compare descriptor "
            "families and test background sensitivity later; these plots do not prove causation.",
            "",
            "## Review and evaluation limits",
            "",
            "Exact-content exclusions, candidate pairs, and review decisions are committed with "
            "the manifest. Uncertain pairs are grouped conservatively. The split code checks that "
            "groups, byte hashes, and decoded-content hashes do not cross partitions.",
            "",
            "Perceptual hashing proposes candidates, not proof of identity. Same-item views with "
            "large viewpoint, rotation, or framing changes can remain undetected. A broader "
            "object-identity review would strengthen the evaluation. Complete item-level isolation "
            "is not established.",
            "",
            "The local sample grid is in `artifacts/eda/training-samples.jpg`. Source-image review "
            "sheets are local in `artifacts/duplicate-review/`. Neither is uploaded to Git.",
            "",
            "No models have been trained or test prediction scores examined at this milestone.",
            "",
        ]
    )
    (reports / "README.md").write_text("\n".join(lines))
    print(f"Wrote training EDA to {reports}")
