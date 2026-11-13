"""External robustness set: photo audit and a single evaluation with the frozen Phase 2 models.

The robustness set is a small collection of everyday waste photos taken by the team on real
backgrounds (table, floor, cloth, grass) under real lighting with several phones. It exists to
measure distribution shift away from TrashNet's white posterboard. It is an external test set:

* it is never used for training, calibration, threshold selection or model selection;
* it is scored exactly once with the three models frozen in
  ``docs/phase-2/evaluation/protocol.json`` (through ``predict_phase2``, so the frozen
  temperatures and demo thresholds are applied);
* ``protocol.json`` and the TrashNet test results are only ever read.

Two entry points:

* ``waste-data robustness-audit`` -> :func:`run_robustness_audit` writes
  ``docs/robustness/manifest.csv`` (one row per photo, with exclusions) and a summary.
* ``waste-phase2-eval robustness`` -> :func:`run_robustness_evaluation` scores the included
  photos once, saves ``robustness-predictions.csv`` and refuses to run again once it exists.
  ``waste-phase2-eval robustness-report`` recomputes the metrics and documents from that saved
  file without scoring anything.

Heavy imports (PyTorch, the models) are deferred so the audit stays light. Every path is
resolved under ``root`` because CI installs the package non-editable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from waste_sorting.acquisition import CLASSES, sha256

# ---------------------------------------------------------------------------
# Paths and constants (all relative to the repository root)
# ---------------------------------------------------------------------------

ROBUSTNESS_IMAGE_ROOT = Path("data/raw/robustness")
TRASHNET_IMAGE_ROOT = Path("data/raw/trashnet")
TRASHNET_MANIFEST = Path("docs/dataset/manifest.csv")
TRASHNET_AUDIT_SETTINGS = Path("docs/dataset/audit_settings.json")
ROBUSTNESS_DOCS = Path("docs/robustness")
PROTOCOL_PATH = Path("docs/phase-2/evaluation/protocol.json")
TEST_METRICS_PATH = Path("docs/phase-2/evaluation/test-metrics.json")
DEFAULT_OUTPUT = Path("docs/phase-2/robustness")
LATEX_PATH = Path("report/robustness-section.tex")

MANIFEST_NAME = "manifest.csv"
CONDITIONS_NAME = "conditions.csv"
AUDIT_SUMMARY_NAME = "audit-summary.json"
DUPLICATE_FLAGS_NAME = "duplicate-flags.csv"
PREDICTIONS_NAME = "robustness-predictions.csv"
METRICS_NAME = "robustness-metrics.json"

MODEL_NAMES = ("phase1_svm", "a2_1_cnn", "h3_hybrid")
DISPLAY_NAMES = {"phase1_svm": "Phase 1 SVM", "a2_1_cnn": "A2.1 CNN", "h3_hybrid": "H3 Hybrid"}
DEFAULT_MCNEMAR_PAIRS = (
    ("h3_hybrid", "a2_1_cnn"),
    ("h3_hybrid", "phase1_svm"),
    ("a2_1_cnn", "phase1_svm"),
)

MANIFEST_COLUMNS = (
    "image_id",
    "label",
    "sha256",
    "phash",
    "width",
    "height",
    "background",
    "lighting",
    "device",
    "included",
    "exclusion_reason",
)
CONDITION_FIELDS = ("background", "lighting", "device")
CONDITIONS_COLUMNS = ("image_id", *CONDITION_FIELDS)
ALLOWED_CONDITIONS = {
    "background": ("table", "floor", "cloth", "grass"),
    "lighting": ("daylight", "indoor", "dim"),
}
DUPLICATE_FLAG_COLUMNS = ("image_id", "matched_id", "matched_source", "phash_distance", "kind")
MIN_PER_CLASS = 50
MIN_TOTAL = 300
SMALL_GROUP = 10
GPS_IFD_TAG = 0x8825

# Exclusion codes, in order of precedence.
UNREADABLE = "unreadable"
GPS_EXIF = "gps_exif"
EXACT_TRASHNET = "exact_duplicate_trashnet"
NEAR_TRASHNET = "near_duplicate_trashnet"
EXACT_WITHIN = "duplicate_content"
NEAR_WITHIN = "near_duplicate_within_set"

Predictor = Callable[..., dict[str, Any]]


class RobustnessError(ValueError):
    """Raised when the robustness set or its records are not in a state that may be used."""


def _under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
    except ValueError:
        return False
    return True


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def has_gps_exif(image: Image.Image) -> bool:
    """Return True if the image carries a non-empty EXIF GPS block (a location leak)."""
    try:
        exif = image.getexif()
    except Exception:
        # Unreadable EXIF is treated as no GPS block; the file is still opened by the audit.
        return False
    if GPS_IFD_TAG not in exif:
        return False
    try:
        return bool(exif.get_ifd(GPS_IFD_TAG))
    except Exception:
        # A GPS pointer we cannot parse is still a warning.
        return True


def scan_robustness_images(image_root: Path) -> pd.DataFrame:
    """Open every file under ``<image_root>/<class>/`` and record hashes, size and GPS EXIF."""
    if not image_root.is_dir():
        raise RobustnessError(f"Robustness photo folder not found: {image_root}")
    rows = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(image_root).as_posix()
        if len(Path(relative).parts) != 2 or path.parent.name not in CLASSES:
            raise RobustnessError(
                f"Unexpected file {relative}: photos must sit directly in one of "
                f"{', '.join(CLASSES)} under {image_root}"
            )
        row = {
            "image_id": relative,
            "label": path.parent.name,
            "sha256": sha256(path),
            "phash": "",
            "pixel_sha256": "",
            "width": 0,
            "height": 0,
            "gps": False,
            "error": "",
        }
        try:
            with Image.open(path) as source:
                source.verify()
            with Image.open(path) as source:
                row["gps"] = has_gps_exif(source)
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                rgb.load()
            row.update(width=rgb.width, height=rgb.height, phash=str(imagehash.phash(rgb)))
            content = f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
            row["pixel_sha256"] = hashlib.sha256(content).hexdigest()
        except (OSError, ValueError, SyntaxError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    if not rows:
        raise RobustnessError(f"No photos found under {image_root}/<class>/")
    return pd.DataFrame(rows)


def load_trashnet_reference(root: Path) -> tuple[pd.DataFrame, int]:
    """Return TrashNet (image_id, sha256, phash) for every audited image and the pHash limit."""
    settings_path = root / TRASHNET_AUDIT_SETTINGS
    manifest_path = root / TRASHNET_MANIFEST
    if not settings_path.is_file() or not manifest_path.is_file():
        raise RobustnessError(
            f"Phase 1 audit records not found ({TRASHNET_MANIFEST}, {TRASHNET_AUDIT_SETTINGS}); "
            "the TrashNet near-duplicate check cannot run."
        )
    threshold = int(json.loads(settings_path.read_text(encoding="utf-8"))["phash_distance"])
    frame = pd.read_csv(
        manifest_path, dtype={"image_id": str, "sha256": str, "phash": str}, keep_default_na=False
    )
    frame = frame.loc[frame["phash"].str.len() > 0, ["image_id", "sha256", "phash"]]
    return frame.reset_index(drop=True), threshold


def read_conditions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(CONDITIONS_COLUMNS) - set(frame.columns)
    if missing:
        raise RobustnessError(f"{path.name} is missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        dupes = sorted(frame.loc[frame["image_id"].duplicated(), "image_id"])
        raise RobustnessError(f"{path.name} lists these images more than once: {dupes}")
    for field in CONDITION_FIELDS:
        frame[field] = frame[field].str.strip().str.lower()
    return frame[list(CONDITIONS_COLUMNS)]


def condition_problems(row: pd.Series | dict[str, Any]) -> list[str]:
    """Return what is missing or invalid in one image's background/lighting/device."""
    problems = []
    for field in CONDITION_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if not value:
            problems.append(f"{field} missing")
        elif field in ALLOWED_CONDITIONS and value not in ALLOWED_CONDITIONS[field]:
            allowed = "/".join(ALLOWED_CONDITIONS[field])
            problems.append(f"{field}={value!r} not one of {allowed}")
    return problems


def _sync_conditions(path: Path, image_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create the hand-filled conditions template, or append blank rows for new photos only."""
    notes: list[str] = []
    if not path.exists():
        template = pd.DataFrame({"image_id": image_ids})
        for field in CONDITION_FIELDS:
            template[field] = ""
        path.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(path, index=False)
        notes.append(
            f"Created {path.name} with {len(image_ids)} blank rows; fill in background, "
            "lighting and device by hand, then rerun the audit."
        )
        return template, notes
    conditions = read_conditions(path)
    known = set(conditions["image_id"])
    new_ids = [image_id for image_id in image_ids if image_id not in known]
    if new_ids:
        extra = pd.DataFrame({"image_id": new_ids})
        for field in CONDITION_FIELDS:
            extra[field] = ""
        conditions = pd.concat([conditions, extra], ignore_index=True)
        conditions.to_csv(path, index=False)
        notes.append(f"Appended {len(new_ids)} blank rows to {path.name} for new photos.")
    stale = sorted(known - set(image_ids))
    if stale:
        notes.append(
            f"{path.name} has {len(stale)} rows for photos that no longer exist "
            f"(ignored): {', '.join(stale[:5])}{' ...' if len(stale) > 5 else ''}"
        )
    return conditions, notes


def build_robustness_manifest(
    scan: pd.DataFrame,
    conditions: pd.DataFrame,
    trashnet: pd.DataFrame,
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exclusions and attach conditions. Returns (manifest, duplicate flags)."""
    frame = scan.sort_values("image_id").reset_index(drop=True).copy()
    frame["included"] = True
    frame["exclusion_reason"] = ""
    flags: list[dict[str, Any]] = []

    def exclude(index: int, reason: str) -> None:
        frame.loc[index, "included"] = False
        frame.loc[index, "exclusion_reason"] = reason

    for index, row in frame.iterrows():
        if row["error"]:
            exclude(index, UNREADABLE)
        elif row["gps"]:
            exclude(index, GPS_EXIF)

    trash_sha = dict(zip(trashnet["sha256"], trashnet["image_id"], strict=True))
    trash_hashes = [
        (image_id, int(phash, 16))
        for image_id, phash in zip(trashnet["image_id"], trashnet["phash"], strict=True)
    ]
    for index, row in frame[frame["included"]].iterrows():
        if row["sha256"] in trash_sha:
            exclude(index, EXACT_TRASHNET)
            flags.append(
                {
                    "image_id": row["image_id"],
                    "matched_id": trash_sha[row["sha256"]],
                    "matched_source": "trashnet",
                    "phash_distance": 0,
                    "kind": EXACT_TRASHNET,
                }
            )
            continue
        value = int(row["phash"], 16)
        best_id, best_distance = None, threshold + 1
        for trash_id, trash_hash in trash_hashes:
            distance = (value ^ trash_hash).bit_count()
            if distance < best_distance:
                best_id, best_distance = trash_id, distance
        if best_id is not None and best_distance <= threshold:
            exclude(index, NEAR_TRASHNET)
            flags.append(
                {
                    "image_id": row["image_id"],
                    "matched_id": best_id,
                    "matched_source": "trashnet",
                    "phash_distance": best_distance,
                    "kind": NEAR_TRASHNET,
                }
            )

    # Within the set: keep the first photo (by image_id) and flag later look-alikes.
    kept: list[tuple[str, str, str, str]] = []  # (image_id, sha256, pixel_sha256, phash)
    for index, row in frame[frame["included"]].iterrows():
        match = None
        for kept_id, kept_sha, kept_pixels, kept_phash in kept:
            if row["sha256"] == kept_sha or row["pixel_sha256"] == kept_pixels:
                match = (kept_id, 0, EXACT_WITHIN)
                break
            distance = _hamming(row["phash"], kept_phash)
            if distance <= threshold:
                match = (kept_id, distance, NEAR_WITHIN)
                break
        if match is None:
            kept.append((row["image_id"], row["sha256"], row["pixel_sha256"], row["phash"]))
            continue
        exclude(index, match[2])
        flags.append(
            {
                "image_id": row["image_id"],
                "matched_id": match[0],
                "matched_source": "robustness",
                "phash_distance": match[1],
                "kind": match[2],
            }
        )

    by_id = conditions.set_index("image_id") if len(conditions) else None
    for field in CONDITION_FIELDS:
        if by_id is None:
            frame[field] = ""
        else:
            frame[field] = frame["image_id"].map(by_id[field]).fillna("")
    manifest = frame[list(MANIFEST_COLUMNS)].copy()
    manifest["included"] = manifest["included"].astype(bool)
    flag_frame = pd.DataFrame(flags, columns=list(DUPLICATE_FLAG_COLUMNS))
    return manifest, flag_frame


def summarise_audit(manifest: pd.DataFrame, flags: pd.DataFrame, threshold: int) -> dict[str, Any]:
    included = manifest[manifest["included"]]
    per_class = {
        label: {
            "files": int((manifest["label"] == label).sum()),
            "included": int((included["label"] == label).sum()),
        }
        for label in CLASSES
    }
    condition_counts = {
        field: {
            (value or "(blank)"): int(count)
            for value, count in included[field].value_counts().sort_index().items()
        }
        for field in CONDITION_FIELDS
    }
    incomplete = [
        f"{row['image_id']}: {'; '.join(problems)}"
        for _, row in included.iterrows()
        if (problems := condition_problems(row))
    ]
    reasons = manifest.loc[~manifest["included"], "exclusion_reason"].value_counts()
    return {
        "phash_distance": threshold,
        "total_files": len(manifest),
        "included_images": len(included),
        "per_class": per_class,
        "condition_counts_included": condition_counts,
        "exclusions": {str(k): int(v) for k, v in reasons.sort_index().items()},
        "gps_warnings": sorted(manifest.loc[manifest["exclusion_reason"] == GPS_EXIF, "image_id"]),
        "unreadable": sorted(manifest.loc[manifest["exclusion_reason"] == UNREADABLE, "image_id"]),
        "duplicate_flags": flags.to_dict(orient="records"),
        "conditions_complete": not incomplete,
        "condition_problems": incomplete,
        "meets_minimum_counts": bool(
            len(included) >= MIN_TOTAL
            and all(v["included"] >= MIN_PER_CLASS for v in per_class.values())
        ),
        "minimum_counts": {"per_class": MIN_PER_CLASS, "total": MIN_TOTAL},
    }


def run_robustness_audit(
    root: Path | str = ".",
    *,
    image_root: Path | str = ROBUSTNESS_IMAGE_ROOT,
    docs_dir: Path | str = ROBUSTNESS_DOCS,
) -> dict[str, Any]:
    """Audit the robustness photos and write the manifest, duplicate flags and summary."""
    root = Path(root).resolve()
    images = _under_root(root, image_root)
    docs = _under_root(root, docs_dir)
    trashnet_root = root / TRASHNET_IMAGE_ROOT
    if _is_relative_to(images, trashnet_root) or _is_relative_to(trashnet_root, images):
        raise RobustnessError(
            f"Robustness photos ({images}) must live outside the TrashNet image root "
            f"({trashnet_root}) so they can never be picked up for training."
        )
    scan = scan_robustness_images(images)
    trashnet, threshold = load_trashnet_reference(root)
    conditions, notes = _sync_conditions(docs / CONDITIONS_NAME, list(scan["image_id"]))
    manifest, flags = build_robustness_manifest(scan, conditions, trashnet, threshold)

    docs.mkdir(parents=True, exist_ok=True)
    manifest_path = docs / MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)
    flags.to_csv(docs / DUPLICATE_FLAGS_NAME, index=False)
    summary = summarise_audit(manifest, flags, threshold)
    summary["notes"] = notes
    summary["manifest_sha256"] = sha256(manifest_path)
    (docs / AUDIT_SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def format_audit_report(summary: dict[str, Any]) -> str:
    """Human-readable counts table and warnings for the terminal."""
    lines = ["", f"{'class':<10} {'files':>6} {'included':>9} {'excluded':>9}"]
    for label, counts in summary["per_class"].items():
        lines.append(
            f"{label:<10} {counts['files']:>6} {counts['included']:>9} "
            f"{counts['files'] - counts['included']:>9}"
        )
    lines.append(
        f"{'total':<10} {summary['total_files']:>6} {summary['included_images']:>9} "
        f"{summary['total_files'] - summary['included_images']:>9}"
    )
    lines.append("")
    lines.append("Included photos per condition:")
    for field, counts in summary["condition_counts_included"].items():
        text = ", ".join(f"{k} {v}" for k, v in counts.items()) or "(none)"
        lines.append(f"  {field:<10} {text}")
    lines.append("")
    if summary["exclusions"]:
        text = ", ".join(f"{k} {v}" for k, v in summary["exclusions"].items())
        lines.append(f"Excluded: {text}")
    if summary["gps_warnings"]:
        lines.append("WARNING: GPS location in EXIF (excluded; retake with location off):")
        lines.extend(f"  {image_id}" for image_id in summary["gps_warnings"])
    if summary["unreadable"]:
        lines.append("WARNING: unreadable files (export as JPEG and retry):")
        lines.extend(f"  {image_id}" for image_id in summary["unreadable"])
    if summary["duplicate_flags"]:
        lines.append(
            f"WARNING: duplicate flags (pHash distance <= {summary['phash_distance']}; "
            "the first photo is kept, remove or retake the flagged one):"
        )
        for flag in summary["duplicate_flags"]:
            lines.append(
                f"  {flag['image_id']} ~ {flag['matched_source']}:{flag['matched_id']} "
                f"(distance {flag['phash_distance']}, {flag['kind']})"
            )
    if not summary["conditions_complete"]:
        problems = summary["condition_problems"]
        lines.append(f"Conditions incomplete for {len(problems)} included photos, e.g.:")
        lines.extend(f"  {text}" for text in problems[:10])
    if not summary["meets_minimum_counts"]:
        lines.append(
            f"Below target: need >= {MIN_PER_CLASS} included photos per class and "
            f">= {MIN_TOTAL} in total."
        )
    lines.extend(summary.get("notes", []))
    if (
        summary["conditions_complete"]
        and summary["meets_minimum_counts"]
        and not summary["gps_warnings"]
        and not summary["duplicate_flags"]
        and not summary["unreadable"]
    ):
        lines.append("Audit clean.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation (single, frozen)
# ---------------------------------------------------------------------------


def load_robustness_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != MANIFEST_COLUMNS:
        raise RobustnessError(f"{path} must have exactly the columns {list(MANIFEST_COLUMNS)}")
    frame["included"] = _as_bool(frame["included"])
    return frame


def _check_ready_for_evaluation(root: Path, manifest_path: Path, image_root: Path) -> pd.DataFrame:
    """Refuse unless the audited manifest is unchanged, complete and matches the photos."""
    summary_path = manifest_path.parent / AUDIT_SUMMARY_NAME
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RobustnessError("Run `waste-data robustness-audit` before the evaluation.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != summary.get("manifest_sha256"):
        raise RobustnessError(
            "docs/robustness/manifest.csv changed since the audit; rerun the audit, do not edit it."
        )
    manifest = load_robustness_manifest(manifest_path)
    included = manifest[manifest["included"]].reset_index(drop=True)
    if included.empty:
        raise RobustnessError("No included robustness photos to evaluate.")
    problems = [
        f"{row['image_id']}: {'; '.join(p)}"
        for _, row in included.iterrows()
        if (p := condition_problems(row))
    ]
    if problems:
        raise RobustnessError(
            "Background/lighting/device must be filled in for every included photo before the "
            f"single evaluation ({len(problems)} incomplete, e.g. {problems[0]}). Edit "
            "docs/robustness/conditions.csv and rerun the audit."
        )
    missing = sorted(set(CLASSES) - set(included["label"]))
    if missing:
        raise RobustnessError(f"Every class needs included photos; missing: {missing}")
    if included["exclusion_reason"].ne("").any() or manifest.loc[
        ~manifest["included"], "exclusion_reason"
    ].eq("").any():
        raise RobustnessError("Manifest inclusion flags are inconsistent with exclusion reasons.")
    trashnet, _ = load_trashnet_reference(root)
    leaked = sorted(set(included["sha256"]) & set(trashnet["sha256"]))
    if leaked:
        raise RobustnessError(f"Included photos are byte-identical to TrashNet images: {leaked}")
    for row in included.itertuples():
        path = image_root / row.image_id
        if not path.is_file():
            raise RobustnessError(f"Photo listed in the manifest is missing: {row.image_id}")
        if sha256(path) != row.sha256:
            raise RobustnessError(f"Photo changed since the audit: {row.image_id}")
    return included


def _load_photo(path: Path) -> Image.Image:
    """Open a phone photo upright (EXIF orientation applied) as RGB."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return image


def _default_predictor() -> Predictor:
    from waste_sorting.phase2_demo import predict_phase2

    return predict_phase2


def _guard_output_dir(root: Path, out_dir: Path) -> None:
    evaluation_dir = (root / PROTOCOL_PATH).parent
    if out_dir.resolve() == evaluation_dir.resolve() or (out_dir / "protocol.json").exists():
        raise RobustnessError(
            f"Refusing to write robustness results into {out_dir}: it holds the frozen protocol "
            "or TrashNet test results. Use docs/phase-2/robustness."
        )


def run_robustness_evaluation(
    root: Path | str = ".",
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    manifest_path: Path | str = ROBUSTNESS_DOCS / MANIFEST_NAME,
    image_root: Path | str = ROBUSTNESS_IMAGE_ROOT,
    latex_path: Path | str | None = LATEX_PATH,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    """Score the included robustness photos ONCE with the three frozen models.

    Refuses to run if ``robustness-predictions.csv`` already exists. Nothing is fitted, tuned,
    recalibrated or reselected: probabilities, temperatures and demo thresholds all come from
    ``predict_phase2`` and the frozen ``protocol.json``, which is only read.
    """
    root = Path(root).resolve()
    out_dir = _under_root(root, output_dir)
    pred_file = out_dir / PREDICTIONS_NAME
    if pred_file.exists():
        raise FileExistsError(
            f"{pred_file} already exists; the robustness set is scored once only. Refusing to "
            "rerun. Use `waste-phase2-eval robustness-report` to rebuild the documents."
        )
    _guard_output_dir(root, out_dir)
    protocol_file = root / PROTOCOL_PATH
    if not protocol_file.is_file():
        raise RobustnessError(f"Frozen protocol not found at {protocol_file}")
    protocol_hash = sha256(protocol_file)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    classes = tuple(protocol["classes"])
    if classes != CLASSES:
        raise RobustnessError(f"Protocol classes {classes} differ from {CLASSES}")

    images = _under_root(root, image_root)
    included = _check_ready_for_evaluation(root, _under_root(root, manifest_path), images)
    predict = predictor or _default_predictor()

    records: list[dict[str, Any]] = []
    for row in included.itertuples():
        photo = _load_photo(images / row.image_id)
        true_idx = classes.index(row.label)
        record: dict[str, Any] = {
            "image_id": row.image_id,
            "label": row.label,
            "label_index": true_idx,
            "background": row.background,
            "lighting": row.lighting,
            "device": row.device,
        }
        for model_name in MODEL_NAMES:
            result = predict(model_name, photo.copy(), root=root)
            frozen_threshold = float(protocol["models"][model_name]["demo_threshold"])
            if not np.isclose(float(result["threshold"]), frozen_threshold):
                raise RobustnessError(
                    f"{model_name} used threshold {result['threshold']}, but protocol.json "
                    f"freezes {frozen_threshold}."
                )
            probs = np.array([result["probabilities"][c] for c in classes], dtype=np.float64)
            pred_idx = int(np.argmax(probs))
            for c_idx, c_name in enumerate(classes):
                record[f"{model_name}_prob_{c_name}"] = float(probs[c_idx])
            record[f"{model_name}_prediction"] = classes[pred_idx]
            record[f"{model_name}_prediction_index"] = pred_idx
            record[f"{model_name}_max_prob"] = float(probs[pred_idx])
            record[f"{model_name}_demo_threshold"] = frozen_threshold
            record[f"{model_name}_correct"] = bool(pred_idx == true_idx)
            record[f"{model_name}_abstained_demo"] = bool(result["abstained"])
        records.append(record)

    if sha256(protocol_file) != protocol_hash:
        raise RobustnessError("protocol.json changed during the robustness evaluation.")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a concurrent or repeated run can never overwrite saved predictions.
    with pred_file.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(records).to_csv(stream, index=False)
    return evaluate_robustness_from_predictions(root, out_dir, latex_path=latex_path)


def _per_condition(df: pd.DataFrame, field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in df.groupby(field, sort=True):
        entry: dict[str, Any] = {
            "n": int(len(group)),
            "small_sample": bool(len(group) < SMALL_GROUP),
        }
        for model_name in MODEL_NAMES:
            correct = _as_bool(group[f"{model_name}_correct"])
            abstained = _as_bool(group[f"{model_name}_abstained_demo"])
            accepted = ~abstained
            entry[model_name] = {
                "accuracy": float(correct.mean()),
                "correct": int(correct.sum()),
                "demo_coverage": float(accepted.mean()),
                "demo_accepted_accuracy": (
                    float(correct[accepted].mean()) if accepted.any() else None
                ),
            }
        result[str(value)] = entry
    return result


def evaluate_robustness_from_predictions(
    root: Path | str = ".",
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    latex_path: Path | str | None = LATEX_PATH,
) -> dict[str, Any]:
    """Compute every robustness metric from the saved predictions; scores nothing."""
    import matplotlib

    matplotlib.use("Agg")
    from waste_sorting.evaluation import compute_metrics, plot_confusion_matrix
    from waste_sorting.phase2_evaluation import compute_bootstrap_ci, compute_pairwise_mcnemar

    root = Path(root).resolve()
    out_dir = _under_root(root, output_dir)
    _guard_output_dir(root, out_dir)
    pred_file = out_dir / PREDICTIONS_NAME
    if not pred_file.is_file():
        raise RobustnessError(f"No saved robustness predictions at {pred_file}")
    protocol = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    test_metrics = json.loads((root / TEST_METRICS_PATH).read_text(encoding="utf-8"))
    classes = tuple(protocol["classes"])
    df = pd.read_csv(pred_file, dtype={"image_id": str, "device": str}, keep_default_na=False)
    y_true = df["label_index"].to_numpy(dtype=np.int64)

    planned = protocol.get("planned_test_analyses", {})
    boot = planned.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 2000))
    boot_seed = int(boot.get("seed", 42))
    pairs = [tuple(p) for p in planned.get("pairwise_mcnemar", {}).get("pairs", [])] or list(
        DEFAULT_MCNEMAR_PAIRS
    )

    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        demo_t = float(protocol["models"][model_name]["demo_threshold"])
        probs = df[[f"{model_name}_prob_{c}" for c in classes]].to_numpy(dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = np.argmax(probs, axis=1)
        preds[model_name] = y_pred
        metrics = compute_metrics(y_true, probs, classes, threshold=demo_t)
        unconditional = metrics["unconditional"]
        selective = metrics["abstention"]
        ci = compute_bootstrap_ci(y_true, y_pred, n_resamples=n_resamples, seed=boot_seed)
        plot_confusion_matrix(
            np.array(unconditional["confusion_matrix"]),
            classes,
            out_dir / f"robustness-confusion-{model_name}.png",
            title=f"Robustness Confusion Matrix: {DISPLAY_NAMES[model_name]}",
        )

        reference = test_metrics["models"][model_name]
        ref_unc = reference["unconditional"]
        ref_demo = reference["selective_demo"]
        ref_ci = reference.get("bootstrap_95_ci", {})
        per_class_recall = {c: float(unconditional["per_class"][c]["recall"]) for c in classes}
        ref_recall = {c: float(ref_unc["per_class"][c]["recall"]) for c in classes}
        models[model_name] = {
            "temperature": float(protocol["models"][model_name]["temperature"]),
            "demo_threshold": demo_t,
            "macro_f1": float(unconditional["macro_f1"]),
            "accuracy": float(unconditional["accuracy"]),
            "expected_calibration_error": float(unconditional["expected_calibration_error"]),
            "brier_score": float(unconditional["brier_score"]),
            "log_loss": float(unconditional["log_loss"]),
            "per_class": unconditional["per_class"],
            "per_class_recall": per_class_recall,
            "confusion_matrix": unconditional["confusion_matrix"],
            "demo_coverage": float(selective["coverage"]),
            "demo_accepted_accuracy": selective.get("accepted_accuracy"),
            "demo_accepted_count": int(selective["accepted_count"]),
            "bootstrap_95_ci": ci,
            "trashnet_test": {
                "sample_count": int(test_metrics["sample_count"]),
                "macro_f1": float(ref_unc["macro_f1"]),
                "accuracy": float(ref_unc["accuracy"]),
                "macro_f1_ci": ref_ci.get("macro_f1"),
                "accuracy_ci": ref_ci.get("accuracy"),
                "demo_coverage": float(ref_demo["coverage"]),
                "demo_accepted_accuracy": float(ref_demo["accepted_accuracy"]),
                "per_class_recall": ref_recall,
            },
            "change_vs_trashnet": {
                "macro_f1": float(unconditional["macro_f1"]) - float(ref_unc["macro_f1"]),
                "accuracy": float(unconditional["accuracy"]) - float(ref_unc["accuracy"]),
                "demo_coverage": float(selective["coverage"]) - float(ref_demo["coverage"]),
                "per_class_recall": {c: per_class_recall[c] - ref_recall[c] for c in classes},
            },
        }

    mcnemar: dict[str, Any] = {}
    for model_a, model_b in pairs:
        mcnemar[f"{model_a}_vs_{model_b}"] = {
            "model_a": model_a,
            "model_b": model_b,
            **compute_pairwise_mcnemar(
                y_true, preds[model_a], preds[model_b], bonferroni_comparisons=len(pairs)
            ),
        }

    result = {
        "sample_count": int(len(df)),
        "classes": list(classes),
        "class_counts": {c: int((df["label"] == c).sum()) for c in classes},
        "predictions_csv": PREDICTIONS_NAME,
        "predictions_sha256": sha256(pred_file),
        "protocol_sha256": sha256(root / PROTOCOL_PATH),
        "test_metrics_sha256": sha256(root / TEST_METRICS_PATH),
        "bootstrap": {"n_resamples": n_resamples, "seed": boot_seed},
        "models": models,
        "mcnemar": mcnemar,
        "by_condition": {field: _per_condition(df, field) for field in CONDITION_FIELDS},
        "note": (
            "External robustness set scored once with the frozen Phase 2 models; nothing was "
            "tuned, recalibrated or reselected on it."
        ),
    }
    manifest_path = root / ROBUSTNESS_DOCS / MANIFEST_NAME
    if manifest_path.is_file():
        result["manifest_sha256"] = sha256(manifest_path)
    (out_dir / METRICS_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(render_robustness_readme(result), encoding="utf-8")
    if latex_path is not None:
        tex = _under_root(root, latex_path)
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_text(render_robustness_latex(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Rendering (all numbers come from robustness-metrics.json)
# ---------------------------------------------------------------------------

VIVA_NOTES = """## Viva notes

1. **Q: Why does an external set matter when we already have a held-out test split?**
   **A:** The TrashNet test split comes from the same distribution as training: one
   photographer, a white posterboard, studio-like lighting. A held-out split measures how
   well the model generalises *within* that distribution. Everyday photos on tables, floors,
   cloth and grass under daylight, indoor and dim light, taken with different phones, are a
   distribution shift (covariate shift in background, illumination and sensor). Only an
   external set shows how much of the TrashNet accuracy survives that shift.
2. **Q: Why must the robustness set never be used for tuning?**
   **A:** As soon as a choice (a threshold, a temperature, a model, an augmentation) is made
   because it scores better on these photos, the photos stop being an unbiased estimate of
   performance on new everyday photos and the reported number becomes optimistic. It would
   also contradict the frozen protocol: models, temperatures and thresholds were committed in
   `protocol.json` before any test or robustness data was scored. That is why the command
   scores the set once and refuses to run again.
3. **Q: If accuracy drops, why not retrain with these photos?**
   **A:** That would be a new Phase 3 experiment needing its own fresh external set. Using
   these photos for training would leave no untouched everyday-photo estimate at all.
4. **Q: Are the condition breakdowns conclusive?**
   **A:** No. Groups are small and conditions are correlated (for example dim light may
   mostly be indoors on one phone), so they show where errors concentrate, not causes.
5. **Q: Does abstention detect out-of-distribution photos?**
   **A:** Not reliably. Coverage at the frozen demo threshold shows how often the models
   became unsure; a confident wrong answer on a new background is still possible.
"""

LIMITS = """## Limitations

- **Small set.** A few hundred photos, about 50 per class, so confidence intervals are wide
  and per-condition cells smaller still; cells under 10 photos are marked as small.
- **Collected by students.** Photos were taken by the team at home and on campus, not a
  random sample of real waste streams; backgrounds, items and phones reflect what we had.
- **Visual material labels.** Labels are the visible main material (TrashNet's six classes),
  assigned by the photographer; they are not local recycling rules, and mixed or dirty items
  were avoided rather than tested.
- **Unpaired comparison.** TrashNet test and robustness images are different sets, so the
  change in accuracy is a difference of two estimates, not a paired test.
- **Correlated conditions.** Background, lighting and phone were not fully crossed.
"""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _ci(ci: dict[str, Any] | None, *, percent: bool) -> str:
    if not ci:
        return ""
    if percent:
        return f" [{100 * ci['ci_low']:.1f}, {100 * ci['ci_high']:.1f}]"
    return f" [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def _worst_conditions(result: dict[str, Any], model_name: str) -> list[str]:
    """Conditions whose accuracy is below the model's overall robustness accuracy, worst first."""
    overall = result["models"][model_name]["accuracy"]
    rows = []
    for field, values in result["by_condition"].items():
        for value, entry in values.items():
            accuracy = entry[model_name]["accuracy"]
            if accuracy < overall:
                rows.append((accuracy, field, value, entry["n"], entry["small_sample"]))
    rows.sort()
    return [
        f"{field}={value} ({_pct(acc)}, n={n}{', small' if small else ''})"
        for acc, field, value, n, small in rows
    ]


def render_robustness_readme(result: dict[str, Any]) -> str:
    models = result["models"]
    classes = result["classes"]
    lines = [
        "# Phase 2 robustness evaluation: everyday photos",
        "",
        "Generated by `waste-phase2-eval robustness` from "
        f"[{PREDICTIONS_NAME}]({PREDICTIONS_NAME}) and [{METRICS_NAME}]({METRICS_NAME}).",
        f"The {result['sample_count']} included photos of the "
        "[external robustness set](../../robustness/README.md) were scored **once** with the "
        "three models frozen in [protocol.json](../evaluation/protocol.json), using the frozen "
        "temperatures and demo thresholds. Nothing was tuned, recalibrated or reselected on "
        "these photos.",
        "",
        "## Summary",
        "",
    ]
    for model_name in MODEL_NAMES:
        m = models[model_name]
        change = m["change_vs_trashnet"]
        delta = change["accuracy"]
        direction = "dropped" if delta < 0 else "rose" if delta > 0 else "stayed"
        worst = _worst_conditions(result, model_name)
        sentence = (
            f"- **{DISPLAY_NAMES[model_name]}**: accuracy {direction} from "
            f"{_pct(m['trashnet_test']['accuracy'])} on the TrashNet test split to "
            f"{_pct(m['accuracy'])} ({100 * change['accuracy']:+.1f} points); macro F1 "
            f"{m['trashnet_test']['macro_f1']:.3f} -> {m['macro_f1']:.3f} "
            f"({change['macro_f1']:+.3f})."
        )
        if worst:
            sentence += f" Weakest conditions: {', '.join(worst[:3])}."
        lines.append(sentence)
    best = max(MODEL_NAMES, key=lambda name: models[name]["macro_f1"])
    lines += [
        f"- Highest robustness macro F1: **{DISPLAY_NAMES[best]}** "
        f"({models[best]['macro_f1']:.3f}). See the McNemar table before calling any "
        "difference real.",
        "",
        f"## Overall results (N = {result['sample_count']}, 95% bootstrap CIs, "
        f"{result['bootstrap']['n_resamples']} resamples, seed {result['bootstrap']['seed']})",
        "",
        "| Model | Macro F1 [95% CI] | Accuracy [95% CI] | ECE | Demo threshold | "
        "Coverage | Accepted accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_name in MODEL_NAMES:
        m = models[model_name]
        ci = m["bootstrap_95_ci"]
        lines.append(
            f"| {DISPLAY_NAMES[model_name]}"
            f" | {m['macro_f1']:.4f}{_ci(ci['macro_f1'], percent=False)}"
            f" | {_pct(m['accuracy'])}{_ci(ci['accuracy'], percent=True)}"
            f" | {m['expected_calibration_error']:.4f} | {m['demo_threshold']:.2f}"
            f" | {_pct(m['demo_coverage'])} ({m['demo_accepted_count']}/{result['sample_count']})"
            f" | {_pct(m['demo_accepted_accuracy'])} |"
        )
    lines += [
        "",
        "## Compared with the TrashNet test split",
        "",
        "Unpaired: different images, so these are differences of two estimates.",
        "",
        "| Model | TrashNet macro F1 | Robustness macro F1 | Change | TrashNet accuracy | "
        "Robustness accuracy | Change | Demo coverage (TrashNet -> robustness) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_name in MODEL_NAMES:
        m = models[model_name]
        t = m["trashnet_test"]
        c = m["change_vs_trashnet"]
        lines.append(
            f"| {DISPLAY_NAMES[model_name]} | {t['macro_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{c['macro_f1']:+.4f} | {_pct(t['accuracy'])} | {_pct(m['accuracy'])} | "
            f"{100 * c['accuracy']:+.1f} pts | {_pct(t['demo_coverage'])} -> "
            f"{_pct(m['demo_coverage'])} |"
        )
    lines += [
        "",
        "## Per-class recall (robustness, change vs TrashNet test in brackets)",
        "",
        "| Class | Photos | " + " | ".join(DISPLAY_NAMES[m] for m in MODEL_NAMES) + " |",
        "| :--- | :---: | " + " | ".join(":---:" for _ in MODEL_NAMES) + " |",
    ]
    for c in classes:
        cells = [
            f"{_pct(models[m]['per_class_recall'][c])} "
            f"({100 * models[m]['change_vs_trashnet']['per_class_recall'][c]:+.1f})"
            for m in MODEL_NAMES
        ]
        lines.append(f"| `{c}` | {result['class_counts'][c]} | " + " | ".join(cells) + " |")
    lines += ["", "## Accuracy by condition", ""]
    for field, values in result["by_condition"].items():
        lines += [
            f"### {field.capitalize()}",
            "",
            "| Value | Photos | " + " | ".join(DISPLAY_NAMES[m] for m in MODEL_NAMES) + " |",
            "| :--- | :---: | " + " | ".join(":---:" for _ in MODEL_NAMES) + " |",
        ]
        for value, entry in values.items():
            note = " (small)" if entry["small_sample"] else ""
            cells = [
                f"{_pct(entry[m]['accuracy'])} ({entry[m]['correct']}/{entry['n']})"
                for m in MODEL_NAMES
            ]
            lines.append(f"| {value}{note} | {entry['n']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Pairwise exact McNemar tests",
        "",
        "| Comparison | Both correct | A only | B only | Both wrong | p-value | "
        "Significant (Bonferroni) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for entry in result["mcnemar"].values():
        lines.append(
            f"| {DISPLAY_NAMES[entry['model_a']]} vs {DISPLAY_NAMES[entry['model_b']]} | "
            f"{entry['both_correct']} | {entry['a_only_correct']} | {entry['b_only_correct']} | "
            f"{entry['both_wrong']} | {entry['p_value']:.4f} | "
            f"{entry['significant_bonferroni']} (alpha {entry['bonferroni_alpha']:.4f}) |"
        )
    lines += ["", "## Confusion matrices", ""]
    for model_name in MODEL_NAMES:
        lines.append(
            f"![{DISPLAY_NAMES[model_name]} robustness confusion matrix]"
            f"(robustness-confusion-{model_name}.png)"
        )
    lines += ["", LIMITS, VIVA_NOTES]
    return "\n".join(lines).rstrip() + "\n"


def _tex_pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def render_robustness_latex(result: dict[str, Any]) -> str:
    models = result["models"]
    n = result["sample_count"]
    parts = []
    for model_name in MODEL_NAMES:
        m = models[model_name]
        parts.append(
            f"\\texttt{{{model_name.replace('_', chr(92) + '_')}}} "
            f"{_tex_pct(m['trashnet_test']['accuracy'])}$\\rightarrow${_tex_pct(m['accuracy'])} "
            f"(macro~F1 {m['macro_f1']:.3f})"
        )
    h3 = models["h3_hybrid"]
    worst = _worst_conditions(result, "h3_hybrid")
    worst_text = (
        " Its weakest conditions were " + ", ".join(w.replace("%", "\\%") for w in worst[:2]) + "."
        if worst
        else ""
    )
    return (
        "% Generated by `waste-phase2-eval robustness` from "
        "docs/phase-2/robustness/robustness-metrics.json. Do not edit numbers by hand.\n"
        "\\subsection{External Robustness Set}\n\\label{sec:robustness}\n"
        f"To measure distribution shift away from TrashNet's white posterboard, the team "
        f"took {n} photos of everyday waste items on tables, floors, cloth and grass under "
        "daylight, indoor and dim lighting with several phones. The set was audited for "
        "duplicates (including against TrashNet) and location metadata, and scored exactly "
        "once with the three models frozen in \\texttt{protocol.json}; nothing was tuned on "
        "it. Accuracy changed from the TrashNet test split as follows: "
        + "; ".join(parts)
        + f". At its frozen demo threshold, \\texttt{{h3\\_hybrid}} kept "
        f"{_tex_pct(h3['demo_coverage'])} coverage (TrashNet test "
        f"{_tex_pct(h3['trashnet_test']['demo_coverage'])})."
        + worst_text
        + " The set is small and collected by students, so per-condition results are "
        "indicative only.\n"
    )


# ---------------------------------------------------------------------------
# CLI: `waste-phase2-eval` dispatcher
# ---------------------------------------------------------------------------

ROBUSTNESS_COMMANDS = ("robustness", "robustness-report")


def _print_evaluation_summary(result: dict[str, Any]) -> None:
    print(f"Robustness set: {result['sample_count']} photos scored once.")
    for model_name in MODEL_NAMES:
        m = result["models"][model_name]
        print(
            f"  {DISPLAY_NAMES[model_name]:<12} macro F1 {m['macro_f1']:.4f} "
            f"(TrashNet {m['trashnet_test']['macro_f1']:.4f}), accuracy {_pct(m['accuracy'])} "
            f"(TrashNet {_pct(m['trashnet_test']['accuracy'])}), demo coverage "
            f"{_pct(m['demo_coverage'])}, accepted accuracy {_pct(m['demo_accepted_accuracy'])}"
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``waste-phase2-eval``: robustness commands here, the rest delegated."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] not in ROBUSTNESS_COMMANDS:
        if not args_list or args_list[0] in ("-h", "--help"):
            print(
                "Robustness commands: waste-phase2-eval robustness | robustness-report "
                "(see `waste-phase2-eval robustness --help`).\n"
            )
        from waste_sorting.phase2_evaluation import main as phase2_main

        phase2_main(args_list)
        return

    parser = argparse.ArgumentParser(
        prog="waste-phase2-eval", description="Single frozen evaluation of the robustness set."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("robustness", "Score the included robustness photos once with the frozen models."),
        ("robustness-report", "Rebuild metrics and documents from saved predictions only."),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Results directory")
        cmd.add_argument("--root", default=".", help="Repository root")
        cmd.add_argument("--latex", default=str(LATEX_PATH), help="LaTeX paragraph output path")
        if name == "robustness":
            cmd.add_argument(
                "--manifest", default=str(ROBUSTNESS_DOCS / MANIFEST_NAME), help="Audit manifest"
            )
            cmd.add_argument(
                "--image-root", default=str(ROBUSTNESS_IMAGE_ROOT), help="Robustness photos"
            )
    args = parser.parse_args(args_list)
    if args.command == "robustness":
        result = run_robustness_evaluation(
            root=args.root,
            output_dir=args.output,
            manifest_path=args.manifest,
            image_root=args.image_root,
            latex_path=args.latex,
        )
    else:
        result = evaluate_robustness_from_predictions(
            root=args.root, output_dir=args.output, latex_path=args.latex
        )
    _print_evaluation_summary(result)


if __name__ == "__main__":
    main()
