"""PyTorch data pipeline for Phase 2: frozen splits, train-only augmentation, sampling.

Reads the committed manifest and fold assignments and never re-splits. Every image
goes through the shared Phase 1 ``preprocess_image`` letterbox first; augmentation is
added to training transforms only. Run ``python -m waste_sorting.dl_data`` to print
split sizes and save a train-only augmentation preview.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode, v2

from waste_sorting.preprocessing import preprocess_image

SPLITS = ("train", "validation", "test")


def load_config(path):
    """Read a TOML config such as configs/phase2.toml."""
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(manifest):
    """Return only included rows, sorted by image_id, after Phase 1 isolation checks.

    Accepts a CSV path or an already-loaded DataFrame; excluded rows are always dropped.
    """
    if isinstance(manifest, pd.DataFrame):
        frame = manifest
    else:
        frame = pd.read_csv(manifest, keep_default_na=False)
    required = {"image_id", "label", "included", "split", "duplicate_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs")
    if not frame["included"].isin([True, False]).all():
        raise ValueError("Manifest inclusion flags must be boolean")
    included = frame.loc[frame["included"] == True].sort_values("image_id")  # noqa: E712
    if not included["split"].isin(SPLITS).all():
        raise ValueError("Included images must have a train/validation/test split")
    if (included["duplicate_group"] == "").any():
        raise ValueError("Every included image needs a duplicate-group ID")
    if (included.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Duplicate group crosses split boundaries")
    return included.reset_index(drop=True)


class WasteImageDataset(Dataset):
    """Included images of one split, or explicit IDs from one split (CV folds).

    Items are ``(tensor[3, H, W] float32, label_index, image_id)`` in image-ID order.
    """

    def __init__(self, manifest, image_root, classes, transform, *, split=None, image_ids=None):
        if (split is None) == (image_ids is None):
            raise ValueError("Pass exactly one of split or image_ids")
        rows = load_manifest(manifest)
        if split is not None:
            if split not in SPLITS:
                raise ValueError(f"split must be one of {SPLITS}")
            rows = rows.loc[rows["split"] == split]
        else:
            image_ids = list(image_ids)
            if len(set(image_ids)) != len(image_ids):
                raise ValueError("image_ids contains duplicates")
            rows = rows.loc[rows["image_id"].isin(image_ids)]
            if len(rows) != len(image_ids):
                raise ValueError("image_ids must all be included manifest rows")
            if rows["split"].nunique() > 1:
                raise ValueError("image_ids must come from a single split")
        if rows.empty:
            raise ValueError("No images selected")
        self.classes = tuple(classes)
        unknown = set(rows["label"]) - set(self.classes)
        if unknown:
            raise ValueError(f"Manifest labels not in configured classes: {sorted(unknown)}")
        self.split = rows["split"].iloc[0]
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_ids = tuple(rows["image_id"])
        self.labels = tuple(self.classes.index(label) for label in rows["label"])
        absent = [i for i in self.image_ids if not (self.image_root / i).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} images missing under {self.image_root}, e.g. {absent[0]}"
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(self.image_root / image_id) as image:
            tensor = self.transform(image)
        return tensor, self.labels[index], image_id


def _augmentations(config):
    """Conservative, label-preserving training steps; each one can be switched off."""
    aug = config["dl"]["augmentation"]
    if not aug["enabled"]:
        return []
    prep = config["preprocessing"]
    steps = []
    if aug["random_resized_crop"]:
        steps.append(
            v2.RandomResizedCrop(
                (prep["height"], prep["width"]),
                scale=tuple(aug["crop_scale"]),
                ratio=tuple(aug["crop_ratio"]),
                antialias=True,
            )
        )
    if aug["horizontal_flip"]:
        steps.append(v2.RandomHorizontalFlip(p=aug["flip_probability"]))
    if aug["rotation"]:
        # Exposed corners take the letterbox colour instead of black.
        steps.append(
            v2.RandomRotation(
                aug["rotation_degrees"],
                interpolation=InterpolationMode.BILINEAR,
                fill=tuple(prep["padding_rgb"]),
            )
        )
    if aug["color_jitter"]:
        steps.append(
            v2.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"],
                saturation=aug["saturation"],
                hue=aug["hue"],
            )
        )
    return steps


def build_transforms(config, train):
    """Letterbox -> (train only: augmentation) -> float tensor in [0, 1] -> normalise."""
    data = config["dl"]["data"]
    steps = [partial(preprocess_image, config=config)]
    if train:
        steps += _augmentations(config)
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data["mean"], std=data["std"]),
        v2.ToPureTensor(),
    ]
    return v2.Compose(steps)


def fold_image_ids(fold, folds_path="docs/training/folds.json"):
    """Return ``(train_ids, validation_ids)`` for one frozen Phase 1 CV fold."""
    payload = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    assignments, n_splits = payload["assignments"], payload["n_splits"]
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < n_splits:
        raise ValueError(f"fold must be an integer in [0, {n_splits})")
    if set(assignments.values()) != set(range(n_splits)):
        raise ValueError("Fold file does not assign images to every fold")
    train = sorted(i for i, f in assignments.items() if f != fold)
    validation = sorted(i for i, f in assignments.items() if f == fold)
    return train, validation


def _class_counts(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or (counts == 0).any():
        raise ValueError("Every class needs at least one label and labels must be < num_classes")
    return labels, counts


def class_weights(labels, num_classes):
    """Inverse-frequency loss weights ``n / (k * n_c)`` from TRAINING labels only."""
    labels, counts = _class_counts(labels, num_classes)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def make_balanced_sampler(labels, num_classes, generator=None):
    """WeightedRandomSampler that draws every class with equal probability."""
    labels, counts = _class_counts(labels, num_classes)
    weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
    return WeightedRandomSampler(weights, len(labels), replacement=True, generator=generator)


def seed_worker(worker_id):
    """Seed NumPy/random in each DataLoader worker from its torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_dataloaders(config, root=".", *, fold=None):
    """Return DataLoaders keyed by split.

    ``fold=None``: train/validation/test from the committed manifest split column.
    ``fold=k``: train on the other folds and validate on fold k of folds.json, exactly
    like Phase 1; no test loader is returned. Call ``torch.manual_seed(seed)`` first
    when ``num_workers == 0`` so augmentation in the main process is reproducible.
    """
    root = Path(root)
    paths, data = config["paths"], config["dl"]["data"]
    classes = tuple(config["classes"])
    manifest = load_manifest(root / paths["manifest"])
    image_root = root / paths["image_root"]
    train_tf, eval_tf = build_transforms(config, True), build_transforms(config, False)

    def dataset(transform, **selection):
        return WasteImageDataset(manifest, image_root, classes, transform, **selection)

    if fold is None:
        datasets = {
            "train": dataset(train_tf, split="train"),
            "validation": dataset(eval_tf, split="validation"),
            "test": dataset(eval_tf, split="test"),
        }
    else:
        train_ids, validation_ids = fold_image_ids(fold, root / paths["folds"])
        expected = set(manifest.loc[manifest["split"] == "train", "image_id"])
        if set(train_ids) | set(validation_ids) != expected:
            raise ValueError("Fold assignments do not match the manifest training split")
        datasets = {
            "train": dataset(train_tf, image_ids=train_ids),
            "validation": dataset(eval_tf, image_ids=validation_ids),
        }

    loaders = {}
    for name, ds in datasets.items():
        generator = torch.Generator().manual_seed(data["seed"])
        options = {}
        if name == "train" and data["balanced_sampling"]:
            options["sampler"] = make_balanced_sampler(ds.labels, len(classes), generator)
        else:
            options["shuffle"] = name == "train"
        loaders[name] = DataLoader(
            ds,
            batch_size=data["batch_size"],
            num_workers=data["num_workers"],
            persistent_workers=data["num_workers"] > 0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
            **options,
        )
    return loaders


def save_augmentation_preview(dataset, config, path, variants=5, seed=42):
    """Save one letterboxed train image per class plus augmented variants as a PNG."""
    from matplotlib.figure import Figure

    if dataset.split != "train":
        raise ValueError("Augmentation previews may only use training images")
    data = config["dl"]["data"]
    mean = torch.tensor(data["mean"]).view(3, 1, 1)
    std = torch.tensor(data["std"]).view(3, 1, 1)
    eval_tf, train_tf = build_transforms(config, False), build_transforms(config, True)
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.labels)
    figure = Figure(figsize=(1.6 * (variants + 1), 1.6 * len(dataset.classes)))
    axes = figure.subplots(len(dataset.classes), variants + 1, squeeze=False)

    def show(axis, tensor, title=None):
        axis.imshow((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        axis.set_xticks([])
        axis.set_yticks([])
        if title:
            axis.set_title(title, fontsize=8)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for row, name in enumerate(dataset.classes):
            candidates = np.flatnonzero(labels == row)
            if candidates.size == 0:
                raise ValueError(f"No training image for class {name}")
            image_id = dataset.image_ids[int(rng.choice(candidates))]
            with Image.open(dataset.image_root / image_id) as image:
                show(axes[row, 0], eval_tf(image), "letterboxed" if row == 0 else None)
                axes[row, 0].set_ylabel(name, fontsize=8)
                for column in range(1, variants + 1):
                    show(axes[row, column], train_tf(image), "augmented" if row == 0 else None)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print Phase 2 split sizes and save a train-only augmentation preview."
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--config", default="configs/phase2.toml", help="Path under --root")
    parser.add_argument("--preview", help="Output PNG path (default under paths.artifacts)")
    parser.add_argument("--variants", type=int, default=5, help="Augmented copies per class")
    args = parser.parse_args(argv)
    root = Path(args.root)
    config = load_config(root / args.config)
    loaders = make_dataloaders(config, root)
    classes = config["classes"]
    for split, loader in loaders.items():
        counts = Counter(loader.dataset.labels)
        per_class = ", ".join(f"{name}={counts[i]}" for i, name in enumerate(classes))
        print(f"{split}: {len(loader.dataset)} images ({per_class})")
    preview = args.preview or root / config["paths"]["artifacts"] / "augmentation_preview.png"
    path = save_augmentation_preview(loaders["train"].dataset, config, preview, args.variants)
    print(f"Saved train-only augmentation preview: {path}")


if __name__ == "__main__":
    main()
