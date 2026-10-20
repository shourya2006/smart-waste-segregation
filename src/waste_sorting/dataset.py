"""Image audit, reviewable duplicate detection, and deterministic group splits."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from waste_sorting.acquisition import CLASSES, EXPECTED_COUNTS, sha256


def audit_images(dataset: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(dataset.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(dataset).as_posix()
        label = path.parent.name
        if label not in CLASSES or len(Path(relative).parts) != 2:
            raise ValueError(f"Unexpected dataset file: {relative}")
        row = {
            "image_id": relative,
            "label": label,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "read_status": "ok",
            "error": "",
            "width": 0,
            "height": 0,
            "mode": "",
            "format": "",
            "pixel_sha256": "",
            "phash": "",
            "brightness": np.nan,
            "mean_r": np.nan,
            "mean_g": np.nan,
            "mean_b": np.nan,
        }
        try:
            with Image.open(path) as source:
                source.verify()
            with Image.open(path) as source:
                row.update(mode=source.mode, format=source.format)
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                rgb.load()
            row.update(width=rgb.width, height=rgb.height)
            content = f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
            row["pixel_sha256"] = hashlib.sha256(content).hexdigest()
            row["phash"] = str(imagehash.phash(rgb))
            small = rgb.resize((64, 64))
            row["brightness"] = round(float(np.asarray(small.convert("L")).mean()), 4)
            for channel, mean in zip("rgb", np.asarray(small).mean(axis=(0, 1)), strict=True):
                row[f"mean_{channel}"] = round(float(mean), 4)
        except (OSError, ValueError, SyntaxError) as error:
            row.update(read_status="unreadable", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No images found under {dataset}")
    frame = pd.DataFrame(rows)
    frame["included"] = frame.read_status.eq("ok")
    frame["exclusion_reason"] = np.where(frame.included, "", "unreadable")
    frame["canonical_id"] = frame.image_id
    valid = frame[frame.included]
    for _, group in valid.groupby("pixel_sha256"):
        if len(group) < 2:
            continue
        if group.label.nunique() != 1:
            # No reliable label can be inferred from identical pixels alone.
            frame.loc[group.index, "included"] = False
            frame.loc[group.index, "exclusion_reason"] = "conflicting_labels"
            frame.loc[group.index, "canonical_id"] = group.image_id.min()
            continue
        canonical = group.image_id.min()
        aliases = group.index[group.image_id.ne(canonical)]
        frame.loc[aliases, "included"] = False
        frame.loc[aliases, "exclusion_reason"] = "duplicate_content"
        frame.loc[group.index, "canonical_id"] = canonical
    return frame


def duplicate_candidates(frame: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    if not 0 <= threshold <= 64:
        raise ValueError("pHash distance must be between 0 and 64")
    valid = frame[frame.included].sort_values("image_id")
    records = list(valid[["image_id", "label", "phash"]].itertuples(index=False, name=None))
    hashes = [int(record[2], 16) for record in records]
    pairs = []
    for i, (left, left_label, _) in enumerate(records):
        for j in range(i + 1, len(records)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= threshold:
                right, right_label, _ = records[j]
                pair_id = hashlib.sha256(f"{left}\n{right}".encode()).hexdigest()[:16]
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "left_id": left,
                        "right_id": right,
                        "left_label": left_label,
                        "right_label": right_label,
                        "phash_distance": distance,
                    }
                )
    return pd.DataFrame(
        pairs,
        columns=[
            "pair_id",
            "left_id",
            "right_id",
            "left_label",
            "right_label",
            "phash_distance",
        ],
    )


def apply_reviews(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Require explicit review; uncertain pairs are kept together conservatively."""
    if reviews.pair_id.duplicated().any():
        raise ValueError("Duplicate review IDs")
    if set(reviews.pair_id) != set(candidates.pair_id):
        raise ValueError("Reviews must cover exactly the current candidate pairs")
    valid_decisions = {"same_item", "different_items", "uncertain"}
    if not set(reviews.decision).issubset(valid_decisions):
        raise ValueError("Every candidate needs same_item, different_items, or uncertain review")
    for field in ("reviewer", "reason"):
        if reviews[field].fillna("").str.strip().eq("").any():
            raise ValueError(f"Every review requires a nonempty {field}")
    parents = {name: name for name in frame.image_id}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    for row in frame.itertuples():
        if row.read_status == "ok":
            union(row.image_id, row.canonical_id)
    decisions = reviews.set_index("pair_id").decision.to_dict()
    for row in candidates.itertuples():
        if decisions[row.pair_id] != "different_items":
            union(row.left_id, row.right_id)
    result = frame.copy()
    result["duplicate_group"] = result.image_id.map(lambda name: find(name))
    return result


def assign_splits(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Greedily assign whole groups against per-class targets; seed resolves ties."""
    if set(fractions) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test fractions")
    ratios = np.array(list(fractions.values()), dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Split fractions must be positive and sum to 1")
    valid = frame[frame.included]
    if set(valid.label) != set(CLASSES):
        raise ValueError("All six classes must have usable images")
    counts = valid.label.value_counts().reindex(CLASSES).to_numpy()
    targets = ratios[:, None] * counts
    assigned = np.zeros_like(targets)
    groups = []
    rng = np.random.default_rng(seed)
    for name, group in valid.groupby("duplicate_group", sort=True):
        vector = group.label.value_counts().reindex(CLASSES, fill_value=0).to_numpy()
        groups.append((name, vector, rng.random()))
    groups.sort(key=lambda item: (-item[1].sum(), item[2], item[0]))
    assignments = {}
    for name, vector, _ in groups:
        # Normalize by class totals, keeping minority classes influential.
        costs = (((assigned + vector - targets) ** 2 - (assigned - targets) ** 2) / counts**2).sum(
            axis=1
        )
        choice = int(np.argmin(costs))
        assignments[name] = list(fractions)[choice]
        assigned[choice] += vector
    if (assigned == 0).any():
        raise ValueError("Cannot populate all classes in all splits with these groups")
    result = frame.copy()
    result["split"] = result.duplicate_group.map(assignments).fillna("excluded")
    result.loc[~result.included, "split"] = "excluded"
    for field in ("duplicate_group", "sha256", "pixel_sha256"):
        if result[result.included].groupby(field).split.nunique().max() > 1:
            raise ValueError(f"Split leakage in {field}")
    return result


def audit_summary(frame: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    valid = frame[frame.included]
    observed = frame.label.value_counts().sort_index().to_dict()
    return {
        "total_files": len(frame),
        "readable_files": int(frame.read_status.eq("ok").sum()),
        "unreadable_files": int(frame.read_status.ne("ok").sum()),
        "included_images": len(valid),
        "excluded_duplicate_content": int(frame.exclusion_reason.eq("duplicate_content").sum()),
        "excluded_label_conflicts": int(frame.exclusion_reason.eq("conflicting_labels").sum()),
        "label_conflict_groups": int(
            frame.loc[frame.exclusion_reason.eq("conflicting_labels"), "pixel_sha256"].nunique()
        ),
        "byte_duplicate_excess": int(frame.sha256.duplicated().sum()),
        "observed_class_counts": observed,
        "upstream_class_counts": EXPECTED_COUNTS,
        "matches_upstream_counts": observed == EXPECTED_COUNTS,
        "near_duplicate_candidates": len(candidates),
        "dimensions": dict(
            Counter(f"{r.width}x{r.height}" for r in frame.itertuples() if r.read_status == "ok")
        ),
        "color_modes": frame.loc[frame.read_status.eq("ok"), "mode"].value_counts().to_dict(),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
