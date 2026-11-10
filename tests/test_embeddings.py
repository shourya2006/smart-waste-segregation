"""Unit tests for embedding extraction, caching, and cache invalidation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch

    from waste_sorting import backbones, dl_data, embeddings
except ImportError:
    torch = None

SKIP_REASON = "PyTorch is not installed (pip install -r requirements-dl.lock)"


def _create_synthetic_environment(root: Path, classes=("cardboard", "glass")):
    """Create a self-contained synthetic repository structure."""
    image_dir = root / "data/raw/trashnet"
    manifest_dir = root / "docs/dataset"
    folds_dir = root / "docs/training"
    configs_dir = root / "configs"

    for d in (image_dir, manifest_dir, folds_dir, configs_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    # 4 train, 2 validation, 2 test per class
    for label in classes:
        (image_dir / label).mkdir(exist_ok=True)
        for split, count in (("train", 4), ("validation", 2), ("test", 2)):
            for i in range(count):
                rel_path = f"{label}/{split}_{i}.png"
                img_path = image_dir / rel_path
                pixel_val = 100 if label == "cardboard" else 200
                img_array = np.full((32, 32, 3), pixel_val, dtype=np.uint8)
                Image.fromarray(img_array).save(img_path)
                rows.append(
                    {
                        "image_id": rel_path,
                        "label": label,
                        "included": True,
                        "exclusion_reason": "",
                        "duplicate_group": rel_path,
                        "split": split,
                        "sha256": "fake_sha",
                    }
                )

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    manifest_path = manifest_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    config = {
        "classes": list(classes),
        "paths": {
            "image_root": "data/raw/trashnet",
            "manifest": "docs/dataset/manifest.csv",
            "folds": "docs/training/folds.json",
            "artifacts": "artifacts/phase2",
        },
        "preprocessing": {
            "width": 32,
            "height": 32,
            "preserve_aspect_ratio": True,
            "padding_rgb": [255, 255, 255],
        },
        "dl": {
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "seed": 42,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "balanced_sampling": False,
            },
            "augmentation": {
                "enabled": True,
                "random_resized_crop": True,
                "crop_scale": [0.8, 1.0],
                "crop_ratio": [0.75, 1.3333],
                "horizontal_flip": True,
                "flip_probability": 0.5,
                "rotation": True,
                "rotation_degrees": 10,
                "color_jitter": True,
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.1,
                "hue": 0.0,
            },
        },
    }
    return config


@unittest.skipIf(torch is None, SKIP_REASON)
class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = _create_synthetic_environment(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extraction_determinism_and_no_augmentation(self):
        """Extraction must be deterministic across calls (eval transform only, no augmentations)."""
        # Fix model instance so random initialization doesn't vary between calls
        model, _ = backbones.load_backbone("mobilenet_v2", pretrained=False)

        # Run extraction twice on the training split using the fixed model
        cache1 = embeddings.extract_embeddings(
            "mobilenet_v2",
            self.config,
            root=self.root,
            split="train",
            model=model,
            pretrained=False,
            device="cpu",
            force=True,
        )
        with np.load(cache1, allow_pickle=False) as data1:
            emb1 = data1["embeddings"]

        # Run again with the exact same model
        cache2 = embeddings.extract_embeddings(
            "mobilenet_v2",
            self.config,
            root=self.root,
            split="train",
            model=model,
            pretrained=False,
            device="cpu",
            force=True,
        )
        with np.load(cache2, allow_pickle=False) as data2:
            emb2 = data2["embeddings"]

        # Check exact equality: eval transform has no random augmentation
        np.testing.assert_array_equal(emb1, emb2)
        self.assertEqual(emb1.shape, (8, 1280))  # 4 train * 2 classes = 8
        self.assertEqual(emb1.dtype, np.float32)

    def test_cache_round_trip_preserves_order_and_labels(self):
        """Round trip through cache keeps image_id order and labels aligned with manifest."""
        embeddings.extract_embeddings(
            "efficientnet_b0",
            self.config,
            root=self.root,
            split="validation",
            pretrained=False,
            device="cpu",
        )

        loaded = embeddings.load_embeddings(
            "efficientnet_b0",
            split="validation",
            root=self.root,
            config=self.config,
            expected_pretrained=False,
        )

        manifest = dl_data.load_manifest(self.root / "docs/dataset/manifest.csv")
        val_manifest = manifest.loc[manifest["split"] == "validation"].reset_index(drop=True)

        expected_ids = val_manifest["image_id"].to_numpy()
        expected_labels = np.array(
            [self.config["classes"].index(lbl) for lbl in val_manifest["label"]],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(loaded["image_ids"], expected_ids)
        np.testing.assert_array_equal(loaded["labels"], expected_labels)
        self.assertEqual(loaded["embeddings"].shape, (4, 1280))  # 2 val * 2 classes = 4

        meta = loaded["metadata"]
        self.assertEqual(meta["backbone"], "efficientnet_b0")
        self.assertEqual(meta["embedding_dim"], 1280)
        self.assertEqual(meta["sample_count"], 4)
        self.assertIn("device", meta)
        self.assertIn(meta["device"], ("cpu", "cuda", "mps"))

    def test_stale_cache_rejected_on_manifest_change(self):
        cache_path = embeddings.extract_embeddings(
            "mobilenet_v2",
            self.config,
            root=self.root,
            split="test",
            pretrained=False,
            device="cpu",
        )
        self.assertTrue(cache_path.is_file())

        # Modify manifest
        manifest_path = self.root / "docs/dataset/manifest.csv"
        manifest_content = manifest_path.read_text()
        manifest_path.write_text(manifest_content + "\n# comment\n")

        with self.assertRaises(ValueError) as ctx:
            embeddings.load_embeddings(
                "mobilenet_v2",
                split="test",
                root=self.root,
                config=self.config,
            )
        self.assertIn("STALE", str(ctx.exception))
        self.assertIn("Manifest SHA-256 mismatch", str(ctx.exception))

    def test_stale_cache_rejected_on_config_change(self):
        embeddings.extract_embeddings(
            "mobilenet_v2",
            self.config,
            root=self.root,
            split="test",
            pretrained=False,
            device="cpu",
        )

        altered_config = dict(self.config)
        altered_config["preprocessing"] = dict(self.config["preprocessing"])
        altered_config["preprocessing"]["width"] = 64

        with self.assertRaises(ValueError) as ctx:
            embeddings.load_embeddings(
                "mobilenet_v2",
                split="test",
                root=self.root,
                config=altered_config,
            )
        self.assertIn("Config hash mismatch", str(ctx.exception))

    def test_stale_cache_rejected_on_backbone_mismatch(self):
        embeddings.extract_embeddings(
            "mobilenet_v2",
            self.config,
            root=self.root,
            split="test",
            pretrained=False,
            device="cpu",
        )

        with self.assertRaises(FileNotFoundError):
            embeddings.load_embeddings(
                "efficientnet_b0",
                split="test",
                root=self.root,
                config=self.config,
            )

    def test_missing_cache_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            embeddings.load_embeddings(
                "mobilenet_v2",
                split="train",
                root=self.root,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
