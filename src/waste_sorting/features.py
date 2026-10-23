"""Handcrafted image feature extraction and reproducible feature caching for Phase 1.

Descriptors implemented:
1. HSV Color Histogram: 16 Hue, 8 Saturation, 8 Value bins (32 features),
   separately normalized per channel so each histogram forms an empirical distribution.
2. Histogram of Oriented Gradients (HOG): 9 orientations, 16x16 pixels per cell,
   2x2 cells per block, L2-Hys normalization over standard grayscale luminance
   (1,764 features for 128x128 images).
3. Uniform Local Binary Patterns (LBP): 8 neighbors, radius 1, normalized 10-bin
   histogram (P + 2 bins: labels 0 through P represent the number of set bits in
   uniform patterns having at most two circular transitions, while non-uniform patterns
   share the remaining bin) computed on integer grayscale (10 features).

Descriptors are concatenated in the fixed order: HSV, HOG, LBP, yielding a 1,806-dim
float32 vector under default project settings. Individual descriptors and subsets
can also be extracted for ablation studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

from waste_sorting.acquisition import sha256
from waste_sorting.preprocessing import preprocess_image

DESCRIPTOR_ORDER = ("color", "hog", "lbp")
DESCRIPTOR_ALIASES = {
    "color": "color",
    "hsv": "color",
    "colour": "color",
    "hog": "hog",
    "lbp": "lbp",
}


def _validate_features_config(config: Mapping) -> dict[str, Any]:
    """Validate and return normalized features configuration dictionary."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping containing 'features' and 'preprocessing'")
    features = config.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("config must contain a 'features' section")

    color_cfg = features.get("color", {})
    if not isinstance(color_cfg, Mapping):
        raise ValueError("features.color must be a mapping")
    space = color_cfg.get("space", "HSV")
    if space.upper() != "HSV":
        raise ValueError(f"Only 'HSV' color space is supported in Phase 1, got {space!r}")
    raw_bins = color_cfg.get("bins", [16, 8, 8])
    if not isinstance(raw_bins, (list, tuple)) or len(raw_bins) != 3:
        raise ValueError("features.color.bins must be a 3-element list/tuple of positive integers")
    for b in raw_bins:
        if isinstance(b, bool) or not isinstance(b, Integral) or b <= 0:
            raise ValueError(f"features.color.bins elements must be positive integers, got {b!r}")
    color_bins = [int(b) for b in raw_bins]

    hog_cfg = features.get("hog", {})
    if not isinstance(hog_cfg, Mapping):
        raise ValueError("features.hog must be a mapping")
    orientations = hog_cfg.get("orientations", 9)
    is_int = isinstance(orientations, Integral) and not isinstance(orientations, bool)
    if not is_int or orientations <= 0:
        raise ValueError("features.hog.orientations must be a positive integer")
    raw_ppc = hog_cfg.get("pixels_per_cell", [16, 16])
    if not isinstance(raw_ppc, (list, tuple)) or len(raw_ppc) != 2:
        raise ValueError("features.hog.pixels_per_cell must be a 2-element list of positive ints")
    for v in raw_ppc:
        if isinstance(v, bool) or not isinstance(v, Integral) or v <= 0:
            raise ValueError("features.hog.pixels_per_cell must contain positive integers")
    pixels_per_cell = (int(raw_ppc[0]), int(raw_ppc[1]))

    raw_cpb = hog_cfg.get("cells_per_block", [2, 2])
    if not isinstance(raw_cpb, (list, tuple)) or len(raw_cpb) != 2:
        raise ValueError("features.hog.cells_per_block must be a 2-element list of positive ints")
    for v in raw_cpb:
        if isinstance(v, bool) or not isinstance(v, Integral) or v <= 0:
            raise ValueError("features.hog.cells_per_block must contain positive integers")
    cells_per_block = (int(raw_cpb[0]), int(raw_cpb[1]))

    block_norm = hog_cfg.get("block_norm", "L2-Hys")
    if block_norm not in {"L1", "L1-sqrt", "L2", "L2-Hys"}:
        raise ValueError(f"Invalid features.hog.block_norm: {block_norm!r}")

    lbp_cfg = features.get("lbp", {})
    if not isinstance(lbp_cfg, Mapping):
        raise ValueError("features.lbp must be a mapping")
    neighbors = lbp_cfg.get("neighbors", 8)
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors <= 0:
        raise ValueError("features.lbp.neighbors must be a positive integer")
    radius = lbp_cfg.get("radius", 1)
    if isinstance(radius, bool) or not isinstance(radius, (Integral, Real)) or radius <= 0:
        raise ValueError("features.lbp.radius must be a positive number")
    method = lbp_cfg.get("method", "uniform")
    if method != "uniform":
        raise ValueError(f"Only method='uniform' is supported for Phase 1 LBP, got {method!r}")

    return {
        "color": {"space": "HSV", "bins": color_bins},
        "hog": {
            "orientations": int(orientations),
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
            "block_norm": str(block_norm),
        },
        "lbp": {
            "neighbors": int(neighbors),
            "radius": float(radius),
            "method": str(method),
        },
    }


def _resolve_descriptors(descriptors: Sequence[str] | None = None) -> list[str]:
    """Validate and sort requested descriptor names into canonical fixed order."""
    if descriptors is None:
        return list(DESCRIPTOR_ORDER)
    resolved = set()
    for name in descriptors:
        key = str(name).strip().lower()
        if key not in DESCRIPTOR_ALIASES:
            raise ValueError(
                f"Unknown descriptor {name!r}; choose from {list(DESCRIPTOR_ALIASES.keys())}"
            )
        resolved.add(DESCRIPTOR_ALIASES[key])
    if not resolved:
        raise ValueError("At least one descriptor must be selected")
    return [name for name in DESCRIPTOR_ORDER if name in resolved]


def extract_hsv_features(image: Image.Image, config: Mapping) -> np.ndarray:
    """Extract normalized 1D HSV channel histograms (32 features by default).

    Converts the image to RGB if not already in RGB mode, maps to HSV space in [0, 1],
    and computes 3 independent histograms (16 Hue bins, 8 Saturation bins, 8 Value bins).
    Each channel histogram is L1-normalized to sum to 1.0 (an empirical probability
    distribution over that channel), ensuring illumination and saturation balance.
    """
    settings = _validate_features_config(config)["color"]
    bins = settings["bins"]
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb_arr = np.asarray(rgb)
    hsv = rgb2hsv(rgb_arr)

    h_channel = np.clip(hsv[..., 0], 0.0, 1.0)
    s_channel = np.clip(hsv[..., 1], 0.0, 1.0)
    v_channel = np.clip(hsv[..., 2], 0.0, 1.0)

    h_hist, _ = np.histogram(h_channel, bins=bins[0], range=(0.0, 1.0))
    s_hist, _ = np.histogram(s_channel, bins=bins[1], range=(0.0, 1.0))
    v_hist, _ = np.histogram(v_channel, bins=bins[2], range=(0.0, 1.0))

    h_norm = h_hist.astype(np.float32) / (float(h_hist.sum()) or 1.0)
    s_norm = s_hist.astype(np.float32) / (float(s_hist.sum()) or 1.0)
    v_norm = v_hist.astype(np.float32) / (float(v_hist.sum()) or 1.0)

    features = np.concatenate([h_norm, s_norm, v_norm]).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("HSV feature extraction produced non-finite values")
    return features


def extract_hog_features(image: Image.Image, config: Mapping) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (1,764 features by default).

    Converts RGB to standard luminance grayscale in [0, 1], then applies Dalal-Triggs HOG
    with 9 orientation bins, 16x16 pixel cells, 2x2 cells per block, and L2-Hys normalization
    (L2 norm followed by clipping at 0.2 and L2 renormalization) to suppress local illumination
    variations and gradient burst highlights.
    """
    settings = _validate_features_config(config)["hog"]
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    ppc_row, ppc_col = settings["pixels_per_cell"]
    cpb_row, cpb_col = settings["cells_per_block"]
    cells_row = rgb.height // ppc_row
    cells_col = rgb.width // ppc_col
    blocks_row = cells_row - cpb_row + 1
    blocks_col = cells_col - cpb_col + 1
    if blocks_row <= 0 or blocks_col <= 0:
        raise ValueError(
            f"Image size ({rgb.width}x{rgb.height}, width x height) is too small "
            f"for HOG pixels_per_cell={settings['pixels_per_cell']} (row, col) and "
            f"cells_per_block={settings['cells_per_block']} (row, col)"
        )
    gray = rgb2gray(np.asarray(rgb))

    feat = hog(
        gray,
        orientations=settings["orientations"],
        pixels_per_cell=settings["pixels_per_cell"],
        cells_per_block=settings["cells_per_block"],
        block_norm=settings["block_norm"],
        feature_vector=True,
    ).astype(np.float32)

    if not np.all(np.isfinite(feat)):
        raise ValueError("HOG feature extraction produced non-finite values")
    return feat


def extract_lbp_features(image: Image.Image, config: Mapping) -> np.ndarray:
    """Extract Uniform Local Binary Patterns normalized histogram (10 features by default).

    Converts RGB to standard 8-bit integer grayscale (ITU-R 601-2 luma in [0, 255]),
    avoiding floating point precision artifacts in neighbor comparisons. Evaluates circular
    uniform LBP with 8 neighbors at radius 1. A binary pattern is uniform if it contains
    at most 2 circular bit transitions (0-to-1 or 1-to-0). For P=8 neighbors, labels 0
    through P represent the number of set bits (0 through 8 ones) in uniform patterns,
    while non-uniform patterns share label P+1 (9), totaling 10 bins. The histogram is
    L1-normalized to sum to 1.0.
    """
    settings = _validate_features_config(config)["lbp"]
    neighbors = settings["neighbors"]
    radius = settings["radius"]
    method = settings["method"]

    gray_u8 = np.asarray(image.convert("L"))
    lbp_map = local_binary_pattern(gray_u8, P=neighbors, R=radius, method=method)

    n_bins = neighbors + 2
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins))
    hist_norm = hist.astype(np.float32) / (float(hist.sum()) or 1.0)

    if not np.all(np.isfinite(hist_norm)):
        raise ValueError("LBP feature extraction produced non-finite values")
    return hist_norm


def extract_features(
    image: Image.Image,
    config: Mapping,
    descriptors: Sequence[str] | None = None,
) -> np.ndarray:
    """Extract handcrafted features from an image using shared preprocessing.

    Applies Om's `preprocess_image` to perform aspect-preserving letterboxing, EXIF
    correction, and canvas padding. Then computes and concatenates the requested
    descriptors in the canonical order: HSV, HOG, LBP.

    Under default configuration:
    - HSV: 32 values
    - HOG: 1,764 values
    - LBP: 10 values
    Total: exactly 1,806 1D float32 values.

    Args:
        image: PIL Image (RGB, RGBA, grayscale, or with EXIF tags).
        config: Full project configuration mapping containing 'preprocessing' and 'features'.
        descriptors: Optional subset of descriptors (e.g. ['color'], ['hog'], ['lbp']).
                     Defaults to all three in fixed order.

    Returns:
        1D numpy array of dtype float32 with finite values.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    preprocessed = preprocess_image(image, config)
    chosen = _resolve_descriptors(descriptors)

    parts: list[np.ndarray] = []
    for name in chosen:
        if name == "color":
            parts.append(extract_hsv_features(preprocessed, config))
        elif name == "hog":
            parts.append(extract_hog_features(preprocessed, config))
        elif name == "lbp":
            parts.append(extract_lbp_features(preprocessed, config))

    result = np.concatenate(parts).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature vector contains non-finite values")
    return result


def get_descriptor_boundaries(
    config: Mapping,
    descriptors: Sequence[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Return slice boundaries (start, end) for each selected descriptor."""
    chosen = _resolve_descriptors(descriptors)
    settings = _validate_features_config(config)
    boundaries = {}
    cursor = 0
    for name in chosen:
        if name == "color":
            dim = sum(settings["color"]["bins"])
        elif name == "hog":
            prep = config.get("preprocessing", {})
            w = prep.get("width", 128)
            h = prep.get("height", 128)
            # Scikit-image interprets pixels_per_cell and cells_per_block as
            # (row, col) = (height, width)
            ppc_row, ppc_col = settings["hog"]["pixels_per_cell"]
            cpb_row, cpb_col = settings["hog"]["cells_per_block"]
            cells_row = h // ppc_row
            cells_col = w // ppc_col
            blocks_row = cells_row - cpb_row + 1
            blocks_col = cells_col - cpb_col + 1
            if blocks_row <= 0 or blocks_col <= 0:
                raise ValueError(
                    f"Canvas size ({w}x{h}, width x height) is too small for HOG "
                    f"pixels_per_cell={settings['hog']['pixels_per_cell']} (row, col) and "
                    f"cells_per_block={settings['hog']['cells_per_block']} (row, col)"
                )
            dim = blocks_row * blocks_col * cpb_row * cpb_col * settings["hog"]["orientations"]
        elif name == "lbp":
            dim = settings["lbp"]["neighbors"] + 2
        boundaries[name] = (cursor, cursor + dim)
        cursor += dim
    return boundaries


def get_feature_names(
    config: Mapping,
    descriptors: Sequence[str] | None = None,
) -> list[str]:
    """Generate human-readable names for each feature in the concatenated vector."""
    chosen = _resolve_descriptors(descriptors)
    settings = _validate_features_config(config)
    names: list[str] = []
    for name in chosen:
        if name == "color":
            bins = settings["color"]["bins"]
            names.extend([f"hsv_h_bin{i:02d}" for i in range(bins[0])])
            names.extend([f"hsv_s_bin{i:02d}" for i in range(bins[1])])
            names.extend([f"hsv_v_bin{i:02d}" for i in range(bins[2])])
        elif name == "hog":
            boundaries = get_descriptor_boundaries(config, [name])
            dim = boundaries[name][1] - boundaries[name][0]
            names.extend([f"hog_{i:04d}" for i in range(dim)])
        elif name == "lbp":
            dim = settings["lbp"]["neighbors"] + 2
            names.extend([f"lbp_bin{i:02d}" for i in range(dim)])
    return names


def get_source_fingerprint() -> str:
    """Return SHA-256 fingerprint of the feature extraction and preprocessing sources."""
    hasher = hashlib.sha256()
    features_src = Path(__file__).read_bytes()
    preprocessing_src = (Path(__file__).parent / "preprocessing.py").read_bytes()
    hasher.update(features_src)
    hasher.update(preprocessing_src)
    return hasher.hexdigest()


def compute_config_hash(config: Mapping) -> str:
    """Compute deterministic SHA-256 hash of relevant classes, preprocessing and feature config."""
    classes = config.get("classes", ["cardboard", "glass", "metal", "paper", "plastic", "trash"])
    relevant = {
        "classes": list(str(c) for c in classes),
        "preprocessing": {
            "width": config.get("preprocessing", {}).get("width", 128),
            "height": config.get("preprocessing", {}).get("height", 128),
            "preserve_aspect_ratio": config.get("preprocessing", {}).get(
                "preserve_aspect_ratio", True
            ),
            "padding_rgb": list(
                config.get("preprocessing", {}).get("padding_rgb", [255, 255, 255])
            ),
        },
        "features": _validate_features_config(config),
    }
    canonical = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_cache_stale(
    cache_path: Path,
    root: Path,
    config: Mapping | None = None,
    check_source: bool = True,
) -> tuple[bool, str]:
    """Check if an existing feature cache file is stale.

    Returns (is_stale, reason). If cache is valid, returns (False, "Cache is up to date").
    """
    if not cache_path.exists():
        return True, f"Cache file does not exist: {cache_path}"

    manifest_path = root / "docs/dataset/manifest.csv"
    if not manifest_path.exists():
        return True, f"Dataset manifest not found: {manifest_path}"

    current_manifest_sha256 = sha256(manifest_path)

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "metadata" not in data:
                return True, "Cache missing metadata payload"
            metadata = json.loads(str(data["metadata"]))
    except Exception as exc:
        return True, f"Failed to read cache file: {exc}"

    cached_manifest_sha = metadata.get("manifest_sha256")
    if cached_manifest_sha != current_manifest_sha256:
        reason = (
            f"Manifest SHA-256 mismatch "
            f"(cache: {cached_manifest_sha[:8] if cached_manifest_sha else 'missing'}..., "
            f"current: {current_manifest_sha256[:8]}...)"
        )
        return True, reason

    if check_source:
        cached_source = metadata.get("source_fingerprint")
        current_source = get_source_fingerprint()
        if not cached_source:
            return True, "Cache missing source_fingerprint in metadata"
        if cached_source != current_source:
            reason = (
                f"Source fingerprint mismatch "
                f"(cache: {cached_source[:8]}..., current: {current_source[:8]}...)"
            )
            return True, reason

    if config is not None:
        if not isinstance(config, Mapping):
            return True, "Invalid configuration: config must be a mapping"
        try:
            expected_config_hash = compute_config_hash(config)
        except (TypeError, ValueError) as exc:
            return True, f"Invalid configuration: {exc}"
    else:
        cfg_path = root / "configs/phase1.toml"
        if not cfg_path.is_file():
            return True, f"Configuration file not found: {cfg_path}"
        try:
            resolved_config = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return True, f"Failed to read configuration from {cfg_path}: {exc}"
        if not isinstance(resolved_config, Mapping):
            return True, f"Invalid configuration in {cfg_path}: root must be a mapping"
        try:
            expected_config_hash = compute_config_hash(resolved_config)
        except (TypeError, ValueError) as exc:
            return True, f"Invalid configuration in {cfg_path}: {exc}"

    cached_config_hash = metadata.get("config_hash")
    if cached_config_hash != expected_config_hash:
        reason = (
            f"Configuration hash mismatch "
            f"(cache: {cached_config_hash[:8] if cached_config_hash else 'missing'}..., "
            f"current: {expected_config_hash[:8]}...)"
        )
        return True, reason

    return False, "Cache is up to date"


def build_feature_cache(
    root: Path,
    config_path: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract handcrafted features for all included manifest images and write cache.

    Preserves:
    - Image IDs and paths relative to raw dataset.
    - Labels, class list, and integer class targets.
    - Duplicate-group IDs.
    - Assigned train, validation, and test splits (never regenerated!).
    - Feature names and descriptor boundaries.
    - Manifest SHA-256, config hash, and extractor source fingerprint.

    Excluded images are strictly omitted; any failure on included images aborts immediately.
    """
    root = root.resolve()
    config_file = config_path or root / "configs/phase1.toml"
    config = tomllib.loads(config_file.read_text())

    output_file = output_path or root / "data/processed/features.npz"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not force and output_file.exists():
        stale, reason = is_cache_stale(output_file, root, config)
        if not stale:
            print(f"Feature cache is already up to date: {output_file}")
            with np.load(output_file, allow_pickle=False) as data:
                return json.loads(str(data["metadata"]))

    manifest_file = root / "docs/dataset/manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Audited manifest not found at {manifest_file}")

    manifest_hash = sha256(manifest_file)
    df = pd.read_csv(manifest_file, keep_default_na=False)

    # Filter strictly to included rows; deterministically ordered by image_id
    included_df = df[df["included"] == True].copy()  # noqa: E712
    included_df.sort_values(by="image_id", inplace=True)
    included_df.reset_index(drop=True, inplace=True)

    dataset_root = root / "data/raw/trashnet"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found at {dataset_root}. Run 'waste-data download' first."
        )

    classes = config.get("classes", ["cardboard", "glass", "metal", "paper", "plastic", "trash"])
    class_to_idx = {name: i for i, name in enumerate(classes)}

    for lbl in included_df["label"].unique():
        if lbl not in class_to_idx:
            raise ValueError(f"Encountered unexpected label {lbl!r} not in config classes")

    feature_names = get_feature_names(config)
    boundaries = get_descriptor_boundaries(config)
    dim = len(feature_names)
    num_samples = len(included_df)

    features_matrix = np.empty((num_samples, dim), dtype=np.float32)
    image_ids: list[str] = []
    relative_paths: list[str] = []
    labels: list[str] = []
    targets: list[int] = []
    splits: list[str] = []
    duplicate_groups: list[str] = []

    print(f"Extracting features for {num_samples} included images (dimension={dim})...")

    for idx, row in enumerate(included_df.itertuples()):
        img_path = dataset_root / row.image_id
        if not img_path.is_file():
            raise FileNotFoundError(
                f"Image file missing from dataset: {img_path} (image_id={row.image_id})"
            )
        # Verify file integrity against audited manifest SHA-256
        file_sha = sha256(img_path)
        if file_sha != row.sha256:
            raise ValueError(
                f"Image file checksum mismatch for {row.image_id}: "
                f"expected {row.sha256}, got {file_sha}"
            )

        try:
            with Image.open(img_path) as img:
                vec = extract_features(img, config)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract features for image {row.image_id}: {exc}"
            ) from exc

        features_matrix[idx, :] = vec
        image_ids.append(row.image_id)
        relative_paths.append(str(Path("data/raw/trashnet") / row.image_id))
        labels.append(row.label)
        targets.append(class_to_idx[row.label])
        splits.append(row.split)
        duplicate_groups.append(row.duplicate_group)

        if (idx + 1) % 500 == 0 or (idx + 1) == num_samples:
            print(f"  Processed {idx + 1}/{num_samples} images")

    config_hash = compute_config_hash(config)
    source_fingerprint = get_source_fingerprint()

    split_counts = pd.Series(splits).value_counts().to_dict()
    class_counts = pd.Series(labels).value_counts().to_dict()

    metadata = {
        "manifest_sha256": manifest_hash,
        "manifest_path": "docs/dataset/manifest.csv",
        "config_hash": config_hash,
        "extractor_version": "0.1.0",
        "source_fingerprint": source_fingerprint,
        "feature_dim": dim,
        "sample_count": num_samples,
        "classes": list(classes),
        "class_to_idx": class_to_idx,
        "descriptors": list(DESCRIPTOR_ORDER),
        "descriptor_boundaries": {k: list(v) for k, v in boundaries.items()},
        "split_counts": split_counts,
        "class_counts": class_counts,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    # Save compressed .npz archive and human-readable metadata .json
    # String arrays use native NumPy unicode strings so allow_pickle=False works safely.
    np.savez_compressed(
        output_file,
        features=features_matrix,
        image_ids=np.array(image_ids),
        relative_paths=np.array(relative_paths),
        labels=np.array(labels),
        targets=np.array(targets, dtype=np.int64),
        splits=np.array(splits),
        duplicate_groups=np.array(duplicate_groups),
        classes=np.array(classes),
        feature_names=np.array(feature_names),
        metadata=np.array(json.dumps(metadata)),
    )

    meta_file = output_file.with_suffix(".json")
    meta_file.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Feature cache written to {output_file} (shape={features_matrix.shape})")
    print(f"Metadata written to {meta_file}")
    return metadata


def load_features(
    cache_path: Path | str,
    root: Path | None = None,
    check_stale: bool = True,
    expected_manifest_sha256: str | None = None,
    expected_config: Mapping | None = None,
) -> dict[str, Any]:
    """Load cached features, validating manifest and configuration integrity.

    Returns a structured dictionary containing numpy arrays:
    - 'features': float32 matrix (N, D)
    - 'labels': 1D string array (N,)
    - 'targets': 1D int64 array (N,)
    - 'splits': 1D string array (N,)
    - 'image_ids': 1D string array (N,)
    - 'duplicate_groups': 1D string array (N,)
    - 'classes': list of class names
    - 'feature_names': list of feature names
    - 'boundaries': dict of descriptor name -> (start_idx, end_idx)
    - 'metadata': cache metadata mapping
    """
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    if check_stale:
        if root is not None:
            stale, reason = is_cache_stale(path, root, config=expected_config)
            if stale:
                raise ValueError(f"Cannot load stale feature cache: {reason}")
        elif expected_config is not None:
            if not isinstance(expected_config, Mapping):
                raise ValueError(
                    "Cannot load stale feature cache: "
                    "Invalid configuration: expected_config must be a mapping"
                )
            try:
                expected_hash = compute_config_hash(expected_config)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot load stale feature cache: Invalid configuration: {exc}"
                ) from exc
            with np.load(path, allow_pickle=False) as data:
                if "metadata" not in data:
                    raise ValueError("Cannot load stale feature cache: missing metadata")
                meta = json.loads(str(data["metadata"]))
            cached_hash = meta.get("config_hash")
            if cached_hash != expected_hash:
                reason = (
                    f"Configuration hash mismatch "
                    f"(cache: {cached_hash[:8] if cached_hash else 'missing'}..., "
                    f"current: {expected_hash[:8]}...)"
                )
                raise ValueError(f"Cannot load stale feature cache: {reason}")
            cached_source = meta.get("source_fingerprint")
            current_source = get_source_fingerprint()
            if not cached_source or cached_source != current_source:
                reason = (
                    f"Source fingerprint mismatch "
                    f"(cache: {cached_source[:8] if cached_source else 'missing'}..., "
                    f"current: {current_source[:8]}...)"
                )
                raise ValueError(f"Cannot load stale feature cache: {reason}")

    with np.load(path, allow_pickle=False) as data:
        metadata_raw = str(data["metadata"])
        metadata = json.loads(metadata_raw)

        if expected_manifest_sha256 and metadata.get("manifest_sha256") != expected_manifest_sha256:
            cached_sha = metadata.get("manifest_sha256")
            raise ValueError(
                f"Cache manifest SHA-256 ({cached_sha}) does not match "
                f"expected ({expected_manifest_sha256})"
            )

        features = data["features"]
        labels = data["labels"]
        targets = data["targets"]
        splits = data["splits"]
        image_ids = data["image_ids"]
        relative_paths = data["relative_paths"]
        duplicate_groups = data["duplicate_groups"]
        classes = list(data["classes"])
        feature_names = list(data["feature_names"])
        raw_boundaries = metadata.get("descriptor_boundaries", {})
        boundaries = {k: (v[0], v[1]) for k, v in raw_boundaries.items()}

    return {
        "features": features,
        "labels": labels,
        "targets": targets,
        "splits": splits,
        "image_ids": image_ids,
        "relative_paths": relative_paths,
        "duplicate_groups": duplicate_groups,
        "classes": classes,
        "class_to_idx": {name: i for i, name in enumerate(classes)},
        "feature_names": feature_names,
        "boundaries": boundaries,
        "metadata": metadata,
    }


def main():
    """CLI entrypoint for standalone feature caching."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root path")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config TOML (default: configs/phase1.toml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output .npz (default: data/processed/features.npz)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force cache regeneration even if up to date"
    )
    args = parser.parse_args()
    build_feature_cache(
        root=args.root,
        config_path=args.config,
        output_path=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
