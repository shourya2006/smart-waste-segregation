"""Unit tests for handcrafted feature extraction and reproducible caching."""

import contextlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting.acquisition import sha256
from waste_sorting.data_cli import main as data_cli_main
from waste_sorting.features import (
    build_feature_cache,
    extract_features,
    extract_hog_features,
    extract_hsv_features,
    extract_lbp_features,
    get_descriptor_boundaries,
    get_feature_names,
    is_cache_stale,
    load_features,
)


def default_config(width=128, height=128):
    return {
        "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        "preprocessing": {
            "width": width,
            "height": height,
            "preserve_aspect_ratio": True,
            "padding_rgb": [255, 255, 255],
        },
        "features": {
            "color": {"space": "HSV", "bins": [16, 8, 8]},
            "hog": {
                "orientations": 9,
                "pixels_per_cell": [16, 16],
                "cells_per_block": [2, 2],
                "block_norm": "L2-Hys",
            },
            "lbp": {"neighbors": 8, "radius": 1, "method": "uniform"},
        },
    }


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()

    def test_default_descriptor_dimensions_and_dtype(self):
        img = Image.new("RGB", (128, 128), (120, 150, 180))
        vec = extract_features(img, self.config)

        self.assertEqual(vec.ndim, 1)
        self.assertEqual(vec.shape[0], 1806)
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_individual_descriptor_dimensions(self):
        img = Image.new("RGB", (128, 128), (200, 100, 50))
        hsv = extract_hsv_features(img, self.config)
        hog_feat = extract_hog_features(img, self.config)
        lbp_feat = extract_lbp_features(img, self.config)

        self.assertEqual(hsv.shape, (32,))
        self.assertEqual(hog_feat.shape, (1764,))
        self.assertEqual(lbp_feat.shape, (10,))
        self.assertEqual(hsv.dtype, np.float32)
        self.assertEqual(hog_feat.dtype, np.float32)
        self.assertEqual(lbp_feat.dtype, np.float32)

    def test_fixed_concatenation_ordering(self):
        img = Image.new("RGB", (128, 128), (80, 160, 240))
        full = extract_features(img, self.config)

        hsv = extract_hsv_features(img, self.config)
        hog_feat = extract_hog_features(img, self.config)
        lbp_feat = extract_lbp_features(img, self.config)

        boundaries = get_descriptor_boundaries(self.config)
        self.assertEqual(boundaries["color"], (0, 32))
        self.assertEqual(boundaries["hog"], (32, 1796))
        self.assertEqual(boundaries["lbp"], (1796, 1806))

        np.testing.assert_allclose(full[0:32], hsv, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(full[32:1796], hog_feat, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(full[1796:1806], lbp_feat, rtol=1e-5, atol=1e-6)

    def test_descriptor_subsets_and_aliases(self):
        img = Image.new("RGB", (128, 128), (50, 100, 150))
        color_only = extract_features(img, self.config, descriptors=["color"])
        hsv_alias = extract_features(img, self.config, descriptors=["hsv"])
        self.assertEqual(color_only.shape, (32,))
        np.testing.assert_array_equal(color_only, hsv_alias)

        hog_only = extract_features(img, self.config, descriptors=["hog"])
        self.assertEqual(hog_only.shape, (1764,))

        lbp_only = extract_features(img, self.config, descriptors=["lbp"])
        self.assertEqual(lbp_only.shape, (10,))

        combo = extract_features(img, self.config, descriptors=["lbp", "color"])
        # Canonical order HSV -> LBP is preserved
        self.assertEqual(combo.shape, (42,))
        np.testing.assert_allclose(combo[:32], color_only)
        np.testing.assert_allclose(combo[32:], lbp_only)

    def test_hsv_normalization_sums(self):
        for color in [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            img = Image.new("RGB", (128, 128), color)
            hsv = extract_hsv_features(img, self.config)
            h_sum = float(hsv[:16].sum())
            s_sum = float(hsv[16:24].sum())
            v_sum = float(hsv[24:32].sum())
            self.assertAlmostEqual(h_sum, 1.0, places=5)
            self.assertAlmostEqual(s_sum, 1.0, places=5)
            self.assertAlmostEqual(v_sum, 1.0, places=5)

    def test_lbp_normalization_sum(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        lbp = extract_lbp_features(img, self.config)
        self.assertEqual(lbp.shape, (10,))
        self.assertAlmostEqual(float(lbp.sum()), 1.0, places=5)
        self.assertTrue(np.all(lbp >= 0.0))

    def test_deterministic_and_independent(self):
        img = Image.new("RGB", (64, 48), (100, 200, 150))
        vec1 = extract_features(img, self.config)
        vec2 = extract_features(img, self.config)
        np.testing.assert_array_equal(vec1, vec2)

        # Mutating input does not affect previously extracted vector
        img.putpixel((0, 0), (0, 0, 0))
        vec3 = extract_features(img, self.config)
        self.assertFalse(np.array_equal(vec1, vec3))

    def test_diverse_modes_via_shared_preprocessing(self):
        # Grayscale 'L'
        gray_img = Image.new("L", (80, 60), 128)
        vec_gray = extract_features(gray_img, self.config)
        self.assertEqual(vec_gray.shape, (1806,))
        self.assertTrue(np.all(np.isfinite(vec_gray)))

        # RGBA with transparency
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        vec_rgba = extract_features(rgba_img, self.config)
        self.assertEqual(vec_rgba.shape, (1806,))
        self.assertTrue(np.all(np.isfinite(vec_rgba)))

        # Fully transparent
        transp_img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        vec_transp = extract_features(transp_img, self.config)
        self.assertEqual(vec_transp.shape, (1806,))
        self.assertTrue(np.all(np.isfinite(vec_transp)))

        # EXIF rotated image
        exif_img = Image.new("RGB", (100, 50), "blue")
        exif_img.getexif()[274] = 6
        vec_exif = extract_features(exif_img, self.config)
        self.assertEqual(vec_exif.shape, (1806,))
        self.assertTrue(np.all(np.isfinite(vec_exif)))

    def test_feature_names_matches_dimension(self):
        names = get_feature_names(self.config)
        self.assertEqual(len(names), 1806)
        self.assertEqual(names[0], "hsv_h_bin00")
        self.assertEqual(names[15], "hsv_h_bin15")
        self.assertEqual(names[16], "hsv_s_bin00")
        self.assertEqual(names[23], "hsv_s_bin07")
        self.assertEqual(names[24], "hsv_v_bin00")
        self.assertEqual(names[31], "hsv_v_bin07")
        self.assertEqual(names[32], "hog_0000")
        self.assertEqual(names[1795], "hog_1763")
        self.assertEqual(names[1796], "lbp_bin00")
        self.assertEqual(names[1805], "lbp_bin09")

    def test_invalid_configurations_rejected(self):
        bad_configs = [
            ({"features": {"color": {"space": "RGB"}}}, ValueError, "HSV"),
            ({"features": {"color": {"bins": [16, 8]}}}, ValueError, "3-element"),
            ({"features": {"color": {"bins": [16, -1, 8]}}}, ValueError, "positive"),
            ({"features": {"hog": {"orientations": 0}}}, ValueError, "positive"),
            ({"features": {"hog": {"block_norm": "L3"}}}, ValueError, "block_norm"),
            ({"features": {"lbp": {"neighbors": -1}}}, ValueError, "positive"),
            ({"features": {"lbp": {"method": "default"}}}, ValueError, "uniform"),
        ]
        img = Image.new("RGB", (32, 32))
        for override, err_type, regex in bad_configs:
            cfg = default_config()
            for k, v in override["features"].items():
                cfg["features"][k].update(v)
            with self.subTest(override=override), self.assertRaisesRegex(err_type, regex):
                extract_features(img, cfg)


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        # Setup directory layout
        (self.root / "configs").mkdir()
        (self.root / "docs/dataset").mkdir(parents=True)
        (self.root / "data/raw/trashnet/paper").mkdir(parents=True)
        (self.root / "data/raw/trashnet/cardboard").mkdir(parents=True)
        (self.root / "data/processed").mkdir(parents=True)

        # Real project configuration
        real_cfg = Path("configs/phase1.toml").read_text()
        (self.root / "configs/phase1.toml").write_text(real_cfg)
        self.config = tomllib.loads(real_cfg)

        # Create sample synthetic images
        self.images = [
            ("cardboard/cardboard1.jpg", "cardboard", "train", "group1", True),
            ("cardboard/cardboard2.jpg", "cardboard", "validation", "group2", True),
            ("paper/paper1.jpg", "paper", "test", "group3", True),
            ("paper/paper2.jpg", "paper", "train", "group4", True),
            ("paper/paper_excluded.jpg", "paper", "train", "group_ex", False),
        ]

        rows = []
        rng = np.random.default_rng(123)
        for rel_path, label, split, dup_group, included in self.images:
            abs_path = self.root / "data/raw/trashnet" / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(abs_path)
            rows.append(
                {
                    "image_id": rel_path,
                    "label": label,
                    "bytes": abs_path.stat().st_size,
                    "sha256": sha256(abs_path),
                    "read_status": "ok",
                    "error": "",
                    "width": 32,
                    "height": 32,
                    "mode": "RGB",
                    "format": "JPEG",
                    "pixel_sha256": "dummy",
                    "phash": "dummy",
                    "brightness": 128.0,
                    "mean_r": 128.0,
                    "mean_g": 128.0,
                    "mean_b": 128.0,
                    "included": included,
                    "exclusion_reason": "" if included else "conflicting_labels",
                    "canonical_id": rel_path,
                    "duplicate_group": dup_group,
                    "split": split,
                }
            )

        self.manifest_df = pd.DataFrame(rows)
        self.manifest_path = self.root / "docs/dataset/manifest.csv"
        self.manifest_df.to_csv(self.manifest_path, index=False)

    def test_build_and_load_cache_preserves_metadata_and_splits(self):
        output_path = self.root / "data/processed/features.npz"
        meta = build_feature_cache(self.root, output_path=output_path)
        self.assertEqual(meta["sample_count"], 4)

        self.assertTrue(output_path.is_file())
        self.assertTrue(output_path.with_suffix(".json").is_file())

        cache = load_features(output_path, root=self.root)

        # 4 included images; 1 excluded image omitted
        self.assertEqual(cache["features"].shape, (4, 1806))
        self.assertEqual(cache["features"].dtype, np.float32)
        self.assertEqual(len(cache["image_ids"]), 4)
        self.assertEqual(len(cache["labels"]), 4)
        self.assertEqual(len(cache["targets"]), 4)
        self.assertEqual(len(cache["splits"]), 4)
        self.assertEqual(len(cache["duplicate_groups"]), 4)

        # Excluded image must NOT be present
        self.assertNotIn("paper/paper_excluded.jpg", cache["image_ids"])

        # Preserved splits must match manifest exactly
        included_df = self.manifest_df[self.manifest_df["included"] == True].sort_values(  # noqa: E712
            "image_id"
        )
        np.testing.assert_array_equal(cache["image_ids"], included_df["image_id"].to_numpy())
        np.testing.assert_array_equal(cache["splits"], included_df["split"].to_numpy())
        np.testing.assert_array_equal(cache["labels"], included_df["label"].to_numpy())
        np.testing.assert_array_equal(
            cache["duplicate_groups"], included_df["duplicate_group"].to_numpy()
        )

        # Metadata integrity
        self.assertEqual(cache["metadata"]["manifest_sha256"], sha256(self.manifest_path))
        self.assertEqual(cache["metadata"]["feature_dim"], 1806)
        self.assertEqual(cache["metadata"]["sample_count"], 4)

    def test_stale_cache_detected_on_manifest_change(self):
        output_path = self.root / "data/processed/features.npz"
        build_feature_cache(self.root, output_path=output_path)

        # Verify not stale initially
        stale, msg = is_cache_stale(output_path, self.root, self.config)
        self.assertFalse(stale)
        self.assertIn("up to date", msg)

        # Modify manifest
        with self.manifest_path.open("a") as f:
            f.write("\n")

        stale, msg = is_cache_stale(output_path, self.root, self.config)
        self.assertTrue(stale)
        self.assertIn("Manifest SHA-256 mismatch", msg)

        # Loading with stale check raises ValueError
        with self.assertRaisesRegex(ValueError, "Cannot load stale feature cache"):
            load_features(output_path, root=self.root, check_stale=True)

    def test_stale_cache_detected_on_config_change(self):
        output_path = self.root / "data/processed/features.npz"
        build_feature_cache(self.root, output_path=output_path)

        # Modify configuration (e.g. change HOG orientations)
        altered_config = default_config()
        altered_config["features"]["hog"]["orientations"] = 12

        stale, msg = is_cache_stale(output_path, self.root, altered_config)
        self.assertTrue(stale)
        self.assertIn("Configuration hash mismatch", msg)

    def test_corrupted_image_aborts_extraction(self):
        corrupted = self.root / "data/raw/trashnet/paper/paper1.jpg"
        corrupted.write_bytes(b"corrupted binary data")

        output_path = self.root / "data/processed/features.npz"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            build_feature_cache(self.root, output_path=output_path)

    def test_missing_image_aborts_extraction(self):
        missing = self.root / "data/raw/trashnet/paper/paper1.jpg"
        missing.unlink()

        output_path = self.root / "data/processed/features.npz"
        with self.assertRaisesRegex(FileNotFoundError, "Image file missing"):
            build_feature_cache(self.root, output_path=output_path)

    def test_cli_features_command_and_idempotence(self):
        # First run creates cache
        with patch("sys.argv", ["waste-data", "--root", str(self.root), "features"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                data_cli_main()
            self.assertIn("Feature cache written", buf.getvalue())

        output_path = self.root / "data/processed/features.npz"
        self.assertTrue(output_path.is_file())

        # Second run without --force skips re-computation
        with patch("sys.argv", ["waste-data", "--root", str(self.root), "features"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                data_cli_main()
            self.assertIn("already up to date", buf.getvalue())

        # Third run with --force regenerates
        with patch("sys.argv", ["waste-data", "--root", str(self.root), "features", "--force"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                data_cli_main()
            self.assertIn("Feature cache written", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
