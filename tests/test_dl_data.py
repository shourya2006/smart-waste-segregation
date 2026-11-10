"""Checks for the Phase 2 PyTorch data pipeline using small synthetic images."""

import contextlib
import copy
import io
import json
import tempfile
import tomllib
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torchvision.transforms import v2

    from waste_sorting import dl_data
except ImportError:  # torch/torchvision are the optional "dl" extra
    torch = None

REPO = Path(__file__).resolve().parents[1]
SKIP_REASON = "PyTorch is not installed (pip install -r requirements-dl.lock)"


def _rows(classes):
    """Per class: 6 train, 2 validation, 2 test and 1 excluded image."""
    rows = []
    for label in classes:
        for split, count in (("train", 6), ("validation", 2), ("test", 2), ("", 1)):
            for n in range(count):
                image_id = f"{label}/{split or 'excluded'}-{n}.png"
                rows.append(
                    {
                        "image_id": image_id,
                        "label": label,
                        "included": bool(split),
                        "exclusion_reason": "" if split else "duplicate",
                        "duplicate_group": image_id if split else "",
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


@unittest.skipIf(torch is None, SKIP_REASON)
class DlDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = tomllib.loads((REPO / "configs/phase2.toml").read_text())
        self.config["dl"]["data"].update(batch_size=4, num_workers=0)
        self.classes = self.config["classes"]
        self.manifest = _rows(self.classes)
        rng = np.random.default_rng(0)
        for image_id in self.manifest["image_id"]:
            path = self.root / "data/raw/trashnet" / image_id
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
        (self.root / "docs/dataset").mkdir(parents=True)
        self.manifest.to_csv(self.root / "docs/dataset/manifest.csv", index=False)
        train_ids = sorted(self.manifest.loc[self.manifest["split"] == "train", "image_id"])
        self.folds = {image_id: i % 3 for i, image_id in enumerate(train_ids)}
        (self.root / "docs/training").mkdir(parents=True)
        (self.root / "docs/training/folds.json").write_text(
            json.dumps({"n_splits": 3, "assignments": self.folds})
        )

    def dataset(self, train=False, **selection):
        transform = dl_data.build_transforms(self.config, train)
        return dl_data.WasteImageDataset(
            self.root / "docs/dataset/manifest.csv",
            self.root / "data/raw/trashnet",
            self.classes,
            transform,
            **selection,
        )

    def image(self):
        with Image.open(self.root / "data/raw/trashnet" / self.manifest["image_id"][0]) as image:
            return image.copy()

    def normalised(self, rgb):
        mean, std = self.config["dl"]["data"]["mean"], self.config["dl"]["data"]["std"]
        return [(c / 255 - m) / s for c, m, s in zip(rgb, mean, std, strict=True)]

    def test_split_lengths_and_labels_match_manifest(self):
        for split in dl_data.SPLITS:
            rows = self.manifest.loc[self.manifest["split"] == split].sort_values("image_id")
            ds = self.dataset(split=split)
            self.assertEqual(ds.image_ids, tuple(rows["image_id"]))
            self.assertEqual(ds.labels, tuple(self.classes.index(x) for x in rows["label"]))
            _, label, image_id = ds[0]
            self.assertEqual((label, image_id), (ds.labels[0], ds.image_ids[0]))

    def test_excluded_rows_never_appear_and_splits_are_disjoint(self):
        ids = {split: set(self.dataset(split=split).image_ids) for split in dl_data.SPLITS}
        excluded = set(self.manifest.loc[~self.manifest["included"], "image_id"])
        for split in dl_data.SPLITS:
            self.assertFalse(ids[split] & excluded)
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])
        with self.assertRaisesRegex(ValueError, "included"):
            self.dataset(image_ids=[next(iter(excluded))])
        with self.assertRaisesRegex(ValueError, "single split"):
            self.dataset(image_ids=[next(iter(ids["train"])), next(iter(ids["test"]))])

    def test_manifest_rejects_group_crossing_splits(self):
        frame = self.manifest.copy()
        frame.loc[frame["split"] == "test", "duplicate_group"] = frame.loc[0, "duplicate_group"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            dl_data.load_manifest(frame)

    def test_committed_manifest_split_sizes_match_summary(self):
        manifest = dl_data.load_manifest(REPO / "docs/dataset/manifest.csv")
        summary = json.loads((REPO / "docs/dataset/split_summary.json").read_text())
        counts = manifest.groupby(["split", "label"]).size()
        for split, per_class in summary["counts"].items():
            self.assertEqual({k: int(v) for k, v in counts[split].items()}, per_class)
        self.assertEqual(manifest["split"].value_counts().to_dict(), summary["totals"])

    def test_fold_helper_reproduces_committed_folds(self):
        path = REPO / "docs/training/folds.json"
        payload = json.loads(path.read_text())
        manifest = dl_data.load_manifest(REPO / "docs/dataset/manifest.csv")
        labels = dict(zip(manifest["image_id"], manifest["label"], strict=True))
        train_split = set(manifest.loc[manifest["split"] == "train", "image_id"])
        for fold in range(payload["n_splits"]):
            train, validation = dl_data.fold_image_ids(fold, path)
            expected = {i for i, f in payload["assignments"].items() if f == fold}
            self.assertEqual(set(validation), expected)
            self.assertEqual(set(train), set(payload["assignments"]) - expected)
            self.assertEqual(set(train) | set(validation), train_split)
            counts = Counter(labels[i] for i in validation)
            self.assertEqual(dict(counts), payload["validation_class_counts"][fold])
        for bad in (-1, payload["n_splits"], True, 1.0):
            with self.assertRaises(ValueError):
                dl_data.fold_image_ids(bad, path)

    def test_fold_loaders_partition_training_split(self):
        loaders = dl_data.make_dataloaders(self.config, self.root, fold=1)
        self.assertEqual(set(loaders), {"train", "validation"})
        train = set(loaders["train"].dataset.image_ids)
        validation = set(loaders["validation"].dataset.image_ids)
        self.assertEqual(validation, {i for i, f in self.folds.items() if f == 1})
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(self.folds))

    def test_output_shape_dtype_and_normalisation(self):
        tensor, _, _ = self.dataset(split="train")[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        # A 2:1 red image is letterboxed by preprocess_image with white bands.
        result = dl_data.build_transforms(self.config, False)(Image.new("RGB", (448, 224), "red"))
        self.assertTrue(torch.allclose(result[:, 0, 0], torch.tensor(self.normalised([255] * 3))))
        red = torch.tensor(self.normalised([255, 0, 0]))
        self.assertTrue(torch.allclose(result[:, 112, 112], red))

    def test_eval_transform_is_deterministic(self):
        image = self.image()
        transform = dl_data.build_transforms(self.config, False)
        torch.manual_seed(1)
        first = transform(image)
        torch.manual_seed(2)
        self.assertTrue(torch.equal(first, transform(image)))

    def test_train_transform_is_random_but_seeded(self):
        image = self.image()
        transform = dl_data.build_transforms(self.config, True)
        torch.manual_seed(0)
        first, second = transform(image), transform(image)
        torch.manual_seed(0)
        self.assertTrue(torch.equal(first, transform(image)))
        self.assertFalse(torch.equal(first, second))

    def test_each_augmentation_can_be_switched_off(self):
        steps = {
            "random_resized_crop": v2.RandomResizedCrop,
            "horizontal_flip": v2.RandomHorizontalFlip,
            "rotation": v2.RandomRotation,
            "color_jitter": v2.ColorJitter,
        }
        for switch, kind in steps.items():
            config = copy.deepcopy(self.config)
            config["dl"]["augmentation"][switch] = False
            types = [type(t) for t in dl_data.build_transforms(config, True).transforms]
            self.assertNotIn(kind, types)
            self.assertEqual(len(types), 8)
        config = copy.deepcopy(self.config)
        config["dl"]["augmentation"]["enabled"] = False
        image = self.image()
        train = dl_data.build_transforms(config, True)
        self.assertTrue(torch.equal(train(image), dl_data.build_transforms(config, False)(image)))

    def test_rotation_fills_with_padding_colour(self):
        config = copy.deepcopy(self.config)
        aug = config["dl"]["augmentation"]
        aug.update(random_resized_crop=False, horizontal_flip=False, color_jitter=False)
        torch.manual_seed(1)  # draws about 5 degrees; seed 0 draws almost 0
        result = dl_data.build_transforms(config, True)(Image.new("RGB", (224, 224), "red"))
        white = torch.tensor(self.normalised([255] * 3))
        self.assertTrue(torch.allclose(result[:, 0, 0], white, atol=1e-4))

    def test_class_order_matches_phase1(self):
        phase1 = tomllib.loads((REPO / "configs/phase1.toml").read_text())
        self.assertEqual(self.classes, phase1["classes"])
        self.assertEqual(self.dataset(split="train").classes, tuple(phase1["classes"]))
        self.assertEqual(self.config["preprocessing"]["padding_rgb"], [255, 255, 255])

    def test_class_weights_are_inverse_frequency(self):
        weights = dl_data.class_weights([0, 1, 1, 1], 2)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.allclose(weights, torch.tensor([2.0, 4 / 6])))
        with self.assertRaises(ValueError):
            dl_data.class_weights([0, 0, 1], 3)
        with self.assertRaises(ValueError):
            dl_data.class_weights([0, 1, 2], 2)

    def test_balanced_sampler_draws_classes_roughly_equally(self):
        labels = [0] * 10 + [1] * 100 + [2] * 40
        generator = torch.Generator().manual_seed(42)
        sampler = dl_data.make_balanced_sampler(labels, 3, generator)
        draws = [labels[i] for _ in range(40) for i in sampler]
        counts = np.bincount(draws, minlength=3) / len(draws)
        self.assertTrue(np.allclose(counts, 1 / 3, atol=0.03), counts)

    def test_loaders_are_reproducible(self):
        def first_epoch(config):
            torch.manual_seed(config["dl"]["data"]["seed"])
            batches = list(dl_data.make_dataloaders(config, self.root)["train"])
            return torch.cat([b[0] for b in batches]), sum((list(b[2]) for b in batches), [])

        for workers, balanced in ((0, True), (2, False)):
            config = copy.deepcopy(self.config)
            config["dl"]["data"].update(num_workers=workers, balanced_sampling=balanced)
            tensors, ids = first_epoch(config)
            again_tensors, again_ids = first_epoch(config)
            self.assertEqual(ids, again_ids)
            self.assertTrue(torch.equal(tensors, again_tensors))
            self.assertEqual(len(ids), 36)
            if not balanced:
                self.assertEqual(sorted(ids), sorted(self.dataset(split="train").image_ids))

    def test_preview_uses_training_images_only(self):
        path = self.root / "artifacts/preview.png"
        dl_data.save_augmentation_preview(self.dataset(True, split="train"), self.config, path, 2)
        with Image.open(path) as image:
            self.assertGreater(image.width, 0)
        with self.assertRaisesRegex(ValueError, "training images"):
            dl_data.save_augmentation_preview(
                self.dataset(split="validation"), self.config, path, 2
            )

    def test_cli_prints_split_sizes(self):
        (self.root / "configs").mkdir()
        text = (REPO / "configs/phase2.toml").read_text()
        text = text.replace("num_workers = 2", "num_workers = 0")
        (self.root / "configs/phase2.toml").write_text(text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dl_data.main(["--root", str(self.root), "--variants", "1"])
        self.assertIn("train: 36 images (cardboard=6", output.getvalue())
        self.assertIn("validation: 12 images", output.getvalue())
        self.assertTrue((self.root / "artifacts/phase2/augmentation_preview.png").is_file())


if __name__ == "__main__":
    unittest.main()
