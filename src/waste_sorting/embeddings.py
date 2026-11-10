"""Extract and cache frozen pretrained deep-learning embeddings for Phase 2.

Runs frozen backbones (MobileNetV2 or EfficientNet-B0) with eval transforms (no augmentation)
over the committed dataset splits, storing float32 embeddings, image IDs, and full
provenance metadata into compressed .npz caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version

from waste_sorting.acquisition import sha256
from waste_sorting.backbones import BACKBONE_WEIGHTS, freeze, load_backbone
from waste_sorting.dl_data import (
    SPLITS,
    WasteImageDataset,
    build_transforms,
    load_config,
    load_manifest,
)

CACHE_SUBDIR = "embeddings"


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of relevant DL data and preprocessing config."""
    sub_config = {
        "classes": config.get("classes"),
        "preprocessing": config.get("preprocessing"),
        "dl_data": config.get("dl", {}).get("data"),
    }
    encoded = json.dumps(sub_config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_source_fingerprint(root: Path) -> str:
    """Fingerprint code relevant to backbone loading, transforms, and embedding extraction."""
    hasher = hashlib.sha256()
    modules = ["backbones.py", "embeddings.py", "dl_data.py", "preprocessing.py"]
    for mod_name in sorted(modules):
        file_path = root / "src/waste_sorting" / mod_name
        if file_path.is_file():
            hasher.update(mod_name.encode("utf-8"))
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _get_git_revision(root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception:
        return None


def get_cache_path(root: Path, backbone_name: str, split: str, config: dict[str, Any]) -> Path:
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts/phase2"))
    return root / artifacts_dir / CACHE_SUBDIR / backbone_name / f"{split}.npz"


def is_embedding_cache_stale(
    cache_path: Path,
    root: Path,
    backbone_name: str,
    config: dict[str, Any],
    *,
    expected_pretrained: bool = True,
) -> tuple[bool, str]:
    """Check whether an existing embedding cache matches manifest, code, config, and weights."""
    if not cache_path.is_file():
        return True, f"Cache file does not exist: {cache_path}"

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "metadata" not in data:
                return True, "Cache missing metadata payload"
            meta = json.loads(str(data["metadata"]))
    except Exception as exc:
        return True, f"Failed to read cache file: {exc}"

    manifest_path = root / config.get("paths", {}).get("manifest", "docs/dataset/manifest.csv")
    current_manifest_sha = sha256(manifest_path)
    if meta.get("manifest_sha256") != current_manifest_sha:
        return (
            True,
            f"Manifest SHA-256 mismatch (cached: {meta.get('manifest_sha256')[:8]}..., "
            f"current: {current_manifest_sha[:8]}...)",
        )

    expected_config_hash = _compute_config_hash(config)
    if meta.get("config_hash") != expected_config_hash:
        return (
            True,
            f"Config hash mismatch (cached: {meta.get('config_hash')[:8]}..., "
            f"current: {expected_config_hash[:8]}...)",
        )

    if meta.get("backbone") != backbone_name:
        return (
            True,
            f"Backbone mismatch (cached: {meta.get('backbone')}, requested: {backbone_name})",
        )

    _, default_weights = BACKBONE_WEIGHTS[backbone_name]
    expected_weights_name = (
        str(default_weights) if expected_pretrained else "None (random uninitialized)"
    )
    if meta.get("weights_name") != expected_weights_name:
        return (
            True,
            f"Weights mismatch (cached: {meta.get('weights_name')}, "
            f"expected: {expected_weights_name})",
        )

    current_fingerprint = _get_source_fingerprint(root)
    if meta.get("source_fingerprint") != current_fingerprint:
        return (
            True,
            f"Source fingerprint mismatch (cached: {meta.get('source_fingerprint')[:8]}..., "
            f"current: {current_fingerprint[:8]}...)",
        )

    return False, "Cache is up to date"


def extract_embeddings(
    backbone_name: str,
    config: dict[str, Any],
    root: Path | str = ".",
    split: str = "train",
    *,
    model: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    output_path: Path | None = None,
    force: bool = False,
    pretrained: bool = True,
) -> Path:
    """Extract and cache frozen embeddings for a single split.

    Uses eval transform only (no augmentation). Runs model under torch.no_grad().
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if backbone_name not in BACKBONE_WEIGHTS:
        raise ValueError(f"Unsupported backbone: {backbone_name!r}")

    root_path = Path(root).resolve()
    target_cache_path = output_path or get_cache_path(root_path, backbone_name, split, config)

    if not force and target_cache_path.is_file():
        stale, reason = is_embedding_cache_stale(
            target_cache_path, root_path, backbone_name, config
        )
        if not stale:
            return target_cache_path

    # Device selection: CUDA -> MPS -> CPU
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    device_type = dev.type

    if model is None:
        model, embedding_dim = load_backbone(backbone_name, pretrained=pretrained)
    else:
        embedding_dim = 1280
    freeze(model)
    model.eval()
    model.to(dev)

    # Use eval transform strictly (no augmentation)
    eval_transform = build_transforms(config, train=False)

    paths_cfg = config.get("paths", {})
    manifest_path = root_path / paths_cfg.get("manifest", "docs/dataset/manifest.csv")
    image_root = root_path / paths_cfg.get("image_root", "data/raw/trashnet")
    classes = tuple(config["classes"])

    dataset = WasteImageDataset(
        manifest=manifest_path,
        image_root=image_root,
        classes=classes,
        transform=eval_transform,
        split=split,
    )

    batch_size = config.get("dl", {}).get("data", {}).get("batch_size", 32)
    # Using num_workers=0 or 2
    num_workers = config.get("dl", {}).get("data", {}).get("num_workers", 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_image_ids: list[str] = []

    with torch.no_grad():
        for batch_images, batch_labels, batch_ids in loader:
            batch_images = batch_images.to(dev)
            emb = model(batch_images)
            all_embeddings.append(emb.cpu())
            all_labels.extend(batch_labels.tolist())
            all_image_ids.extend(batch_ids)

    if all_embeddings:
        embeddings_np = torch.cat(all_embeddings, dim=0).to(torch.float32).numpy()
    else:
        embeddings_np = np.empty((0, embedding_dim), dtype=np.float32)

    labels_np = np.array(all_labels, dtype=np.int64)
    image_ids_np = np.array(all_image_ids)

    _, default_weights = BACKBONE_WEIGHTS[backbone_name]
    weights_name = str(default_weights) if pretrained else "None (random uninitialized)"

    prep = config.get("preprocessing", {})
    image_size = (prep.get("height", 224), prep.get("width", 224))

    metadata = {
        "backbone": backbone_name,
        "weights_name": weights_name,
        "embedding_dim": int(embedding_dim),
        "image_size": list(image_size),
        "split": split,
        "sample_count": len(dataset),
        "manifest_sha256": sha256(manifest_path),
        "config_hash": _compute_config_hash(config),
        "source_fingerprint": _get_source_fingerprint(root_path),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "git_revision": _get_git_revision(root_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": device_type,
        "classes": list(classes),
    }

    target_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_cache_path,
        embeddings=embeddings_np,
        labels=labels_np,
        image_ids=image_ids_np,
        metadata=json.dumps(metadata, indent=2),
    )

    return target_cache_path


def load_embeddings(
    backbone_name: str,
    split: str,
    root: Path | str = ".",
    config: dict[str, Any] | None = None,
    *,
    expected_pretrained: bool = True,
) -> dict[str, Any]:
    """Load cached embeddings and verify freshness.

    Refuses stale or missing cache with an informative ValueError.
    Returns dictionary with:
    - embeddings: np.ndarray [N, 1280] float32
    - labels: np.ndarray [N] int64
    - image_ids: np.ndarray [N] str
    - metadata: dict
    """
    root_path = Path(root).resolve()
    if config is None:
        cfg_path = root_path / "configs/phase2.toml"
        config = load_config(cfg_path)

    cache_path = get_cache_path(root_path, backbone_name, split, config)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Embedding cache not found for backbone {backbone_name!r} and split {split!r} "
            f"at {cache_path}. Run 'waste-dl-embeddings --backbone {backbone_name} "
            f"--splits {split}' to extract it."
        )

    stale, reason = is_embedding_cache_stale(
        cache_path,
        root_path,
        backbone_name,
        config,
        expected_pretrained=expected_pretrained,
    )
    if stale:
        raise ValueError(
            f"Embedding cache is STALE: {reason}. "
            f"Please re-extract by running: waste-dl-embeddings --backbone {backbone_name} "
            f"--splits {split}"
        )

    # Double check alignment with manifest
    manifest_path = root_path / config.get("paths", {}).get("manifest", "docs/dataset/manifest.csv")
    manifest = load_manifest(manifest_path)
    split_manifest = manifest.loc[manifest["split"] == split].reset_index(drop=True)
    classes = tuple(config["classes"])
    expected_ids = split_manifest["image_id"].to_numpy()
    expected_labels = np.array(
        [classes.index(lbl) for lbl in split_manifest["label"]],
        dtype=np.int64,
    )

    with np.load(cache_path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
        image_ids = data["image_ids"]
        metadata = json.loads(str(data["metadata"]))

    if not np.array_equal(image_ids, expected_ids):
        raise ValueError(f"Cached image_ids do not match the manifest for split {split!r}.")
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Cached labels do not match the manifest labels for split {split!r}.")

    return {
        "embeddings": embeddings,
        "labels": labels,
        "image_ids": image_ids,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache frozen deep learning embeddings for TrashNet."
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONE_WEIGHTS.keys()),
        required=True,
        help="Pretrained backbone architecture to use",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Splits to extract (default: train validation test)",
    )
    parser.add_argument(
        "--config",
        default="configs/phase2.toml",
        help="Path to Phase 2 TOML configuration file",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Path to repository root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if valid cache exists",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = load_config(root / args.config)

    print(f"Extracting embeddings using backbone: {args.backbone}")
    for split in args.splits:
        t0 = time.perf_counter()
        cache_path = extract_embeddings(
            backbone_name=args.backbone,
            config=config,
            root=root,
            split=split,
            force=args.force,
        )
        elapsed = time.perf_counter() - t0
        # Load and report shape and elapsed time, NO metrics
        data = np.load(cache_path, allow_pickle=False)
        emb_shape = data["embeddings"].shape
        meta = json.loads(str(data["metadata"]))
        dev_info = meta.get("device", "unknown")
        print(
            f"[{args.backbone} - {split} | {dev_info}] Shape: {emb_shape} "
            f"extracted in {elapsed:.2f}s -> {cache_path}"
        )


if __name__ == "__main__":
    main()
