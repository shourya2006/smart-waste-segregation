import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from waste_sorting.acquisition import sha256
from waste_sorting.evaluation import (
    compute_calibration_curve_and_ece,
    fit_calibrated_ensemble,
    load_manifest,
    multiclass_brier_score,
    reconstruct_selected_pipeline,
    run_pipeline,
    select_abstention_threshold,
    sweep_abstention_thresholds,
    validate_feature_cache,
)
from waste_sorting.extended_features import (
    DIMENSION as EXTENDED_DIMENSION,
)
from waste_sorting.extended_features import (
    SETTINGS as EXTENDED_SETTINGS,
)


class EvaluationModuleTests(unittest.TestCase):
    def test_reconstructed_pipeline_parameters_and_structure(self):
        pipe = reconstruct_selected_pipeline(seed=42)
        steps = dict(pipe.named_steps)
        self.assertIn("features", steps)
        self.assertIn("scale", steps)
        self.assertIn("classifier", steps)

        # 1,608 selected columns from 2,760 input dimensions
        selected_cols = steps["features"].transformers[0][2]
        self.assertEqual(len(selected_cols), 1608)

        clf = steps["classifier"]
        self.assertEqual(clf.C, 10.0)
        self.assertEqual(clf.gamma, 10.0)
        self.assertIsNone(clf.class_weight)
        self.assertEqual(clf.random_state, 42)

    def test_multiclass_brier_score_formula_and_bounds(self):
        # Perfect predictions: Brier score must be 0.0
        y_true = np.array([0, 1, 2])
        y_prob_perfect = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertAlmostEqual(multiclass_brier_score(y_true, y_prob_perfect, n_classes=6), 0.0)

        # Worst possible predictions: Brier score must be 2.0
        # (1-0)^2 + (0-1)^2 = 2.0 per row
        y_prob_worst = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # true is 0, predicted 1
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true is 1, predicted 0
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # true is 2, predicted 1
            ]
        )
        self.assertAlmostEqual(multiclass_brier_score(y_true, y_prob_worst, n_classes=6), 2.0)

        # Uniform predictions: (1 - 1/6)^2 + 5 * (1/6)^2 = 25/36 + 5/36 = 30/36 = 5/6
        y_prob_uniform = np.full((3, 6), 1.0 / 6.0)
        expected_uniform_bs = 5.0 / 6.0
        self.assertAlmostEqual(
            multiclass_brier_score(y_true, y_prob_uniform, n_classes=6), expected_uniform_bs
        )

    def test_expected_calibration_error_and_bins(self):
        # Perfectly calibrated synthetic sample: confidences exactly equal accuracies
        y_true = np.array([0, 1, 0, 1])
        # Two samples with conf 1.0, two with conf 0.5 (one correct, one wrong -> avg acc 0.5)
        y_prob = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true 0, pred 0 (acc 1.0, conf 1.0)
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # true 1, pred 1 (acc 1.0, conf 1.0)
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # true 0, pred 0 (acc 1.0, conf 0.5)
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # true 1, pred 0 (acc 0.0, conf 0.5)
            ]
        )
        cal = compute_calibration_curve_and_ece(y_true, y_prob, n_bins=10)
        self.assertEqual(cal["n_bins"], 10)
        self.assertEqual(len(cal["bins"]), 10)
        self.assertAlmostEqual(cal["ece"], 0.0)

        # Verify empty bins handled with 0 weight
        empty_bins = [b for b in cal["bins"] if b["count"] == 0]
        self.assertGreater(len(empty_bins), 0)
        for eb in empty_bins:
            self.assertEqual(eb["accuracy"], 0.0)

    def test_threshold_selection_highest_coverage_with_both_conditions(self):
        # Sweep table where multiple thresholds meet acc >= 0.90 and cov >= 0.80
        data = [
            {"threshold": 0.40, "coverage": 0.90, "accepted_accuracy": 0.88,
             "accepted_count": 90, "abstained_count": 10, "total_count": 100},
            {"threshold": 0.50, "coverage": 0.85, "accepted_accuracy": 0.92,
             "accepted_count": 85, "abstained_count": 15, "total_count": 100},
            {"threshold": 0.55, "coverage": 0.85, "accepted_accuracy": 0.93,
             "accepted_count": 85, "abstained_count": 15, "total_count": 100},
            {"threshold": 0.60, "coverage": 0.82, "accepted_accuracy": 0.95,
             "accepted_count": 82, "abstained_count": 18, "total_count": 100},
        ]
        sweep_df = pd.DataFrame(data)
        res = select_abstention_threshold(sweep_df, target_accuracy=0.90, min_coverage=0.80)

        # Max coverage meeting both criteria is 0.85 (at 0.50 and 0.55)
        # Tie-breaker requires lower threshold -> 0.50
        self.assertTrue(res["target_met"])
        self.assertAlmostEqual(res["threshold"], 0.50)
        self.assertAlmostEqual(res["coverage"], 0.85)
        self.assertAlmostEqual(res["accepted_accuracy"], 0.92)

    def test_threshold_selection_fallback_when_accuracy_target_infeasible(self):
        # No threshold reaches 0.90 accuracy
        data = [
            {"threshold": 0.30, "coverage": 0.95, "accepted_accuracy": 0.78,
             "accepted_count": 95, "abstained_count": 5, "total_count": 100},
            {"threshold": 0.40, "coverage": 0.86, "accepted_accuracy": 0.84,
             "accepted_count": 86, "abstained_count": 14, "total_count": 100},
            {"threshold": 0.45, "coverage": 0.82, "accepted_accuracy": 0.84,
             "accepted_count": 82, "abstained_count": 18, "total_count": 100},
            {"threshold": 0.50, "coverage": 0.75, "accepted_accuracy": 0.91,
             "accepted_count": 75, "abstained_count": 25, "total_count": 100},
        ]
        sweep_df = pd.DataFrame(data)
        res = select_abstention_threshold(sweep_df, target_accuracy=0.90, min_coverage=0.80)

        # Fallback: among coverage >= 0.80, max accuracy is 0.84 (at 0.40 and 0.45).
        # Tied accuracy resolved by higher coverage (0.86 vs 0.82) -> 0.40
        self.assertFalse(res["target_met"])
        self.assertAlmostEqual(res["threshold"], 0.40)
        self.assertAlmostEqual(res["coverage"], 0.86)
        self.assertAlmostEqual(res["accepted_accuracy"], 0.84)
        self.assertIn("not achieved", res["note"])

    def test_boundary_equality_in_threshold_sweep(self):
        y_true = np.array([0, 1])
        # Two predictions with max prob exactly 0.70
        y_prob = np.array(
            [
                [0.70, 0.30, 0.0, 0.0, 0.0, 0.0],
                [0.30, 0.70, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        sweep = sweep_abstention_thresholds(y_true, y_prob)
        row_at_070 = sweep[np.isclose(sweep["threshold"], 0.70)].iloc[0]
        # At tau=0.70, samples with max_prob == 0.70 must be ACCEPTED (>= equality)
        self.assertEqual(row_at_070["accepted_count"], 2)
        self.assertAlmostEqual(row_at_070["coverage"], 1.0)

        row_at_071 = sweep[np.isclose(sweep["threshold"], 0.71)].iloc[0]
        # At tau=0.71, samples with max_prob == 0.70 must be ABSTAINED
        self.assertEqual(row_at_071["accepted_count"], 0)
        self.assertAlmostEqual(row_at_071["coverage"], 0.0)

    def test_manifest_validation_and_split_isolation(self):
        root = Path(__file__).resolve().parent.parent
        manifest = load_manifest(root)
        self.assertEqual(len(manifest), 2521)
        self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})

        # Group isolation
        groups_by_split = manifest.groupby("duplicate_group")["split"].nunique()
        self.assertTrue((groups_by_split == 1).all())

        # Exact split counts
        counts = manifest["split"].value_counts()
        self.assertEqual(counts["train"], 1764)
        self.assertEqual(counts["validation"], 380)
        self.assertEqual(counts["test"], 377)

    def test_training_and_calibration_fold_isolation(self):
        # Test synthetic calibrated ensemble with explicit folds
        np.random.seed(42)
        n_samples = 30
        X_toy = np.abs(np.random.randn(n_samples, 2760).astype(np.float32))
        y_toy = np.tile(np.arange(6), 5)  # 5 samples per class
        toy_folds = [
            (np.setdiff1d(np.arange(n_samples), np.arange(0, 6)), np.arange(0, 6)),
            (np.setdiff1d(np.arange(n_samples), np.arange(6, 12)), np.arange(6, 12)),
            (np.setdiff1d(np.arange(n_samples), np.arange(12, 18)), np.arange(12, 18)),
            (np.setdiff1d(np.arange(n_samples), np.arange(18, 24)), np.arange(18, 24)),
            (np.setdiff1d(np.arange(n_samples), np.arange(24, 30)), np.arange(24, 30)),
        ]

        cal_model = fit_calibrated_ensemble(X_toy, y_toy, toy_folds, seed=42)
        self.assertEqual(len(cal_model.calibrated_classifiers_), 5)

        # Check probability outputs
        probs = cal_model.predict_proba(X_toy)
        self.assertEqual(probs.shape, (n_samples, 6))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_final_evaluation_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "eval_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            guard_file = out_dir / "test-predictions.csv"
            guard_file.write_text("existing evaluation predictions")

            root = Path(__file__).resolve().parent.parent
            art_dir = Path(tmp_dir) / "artifacts"

            with self.assertRaises(FileExistsError) as ctx:
                run_pipeline(root, out_dir, art_dir, allow_overwrite=False)
            self.assertIn("A guard prevents accidentally overwriting", str(ctx.exception))


class FeatureCacheValidationTests(unittest.TestCase):
    """Synthetic tests verifying extended-feature cache validation rules."""

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.split_name = "validation"
        self.n_samples = 8
        self.image_ids = np.array([f"img_{i:03d}.jpg" for i in range(self.n_samples)])
        self.manifest = pd.DataFrame(
            [
                {
                    "image_id": self.image_ids[i],
                    "label": "cardboard",
                    "included": True,
                    "split": self.split_name,
                    "duplicate_group": f"grp_{i}",
                    "sha256": "dummy_checksum",
                }
                for i in range(self.n_samples)
            ]
        )
        self.manifest_sha = sha256(self.root / "docs/dataset/manifest.csv")
        self.extractor_sha = sha256(self.root / "src/waste_sorting/extended_features.py")
        self.settings_hash = hashlib.sha256(
            json.dumps(EXTENDED_SETTINGS, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.valid_features = np.ones(
            (self.n_samples, EXTENDED_DIMENSION), dtype=np.float32
        )

    def _save_cache(self, directory: Path, filename: str = "cache.npz", **kwargs) -> Path:
        payload = {
            "features": self.valid_features,
            "image_ids": self.image_ids,
            "split": self.split_name,
            "manifest_sha256": self.manifest_sha,
            "extractor_sha256": self.extractor_sha,
            "settings_hash": self.settings_hash,
        }
        payload.update(kwargs)
        cache_path = directory / filename
        np.savez_compressed(cache_path, **payload)
        return cache_path

    def test_valid_cache_passes_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = self._save_cache(Path(tmp_dir))
            mat = validate_feature_cache(
                cache_path, self.split_name, self.root, self.manifest
            )
            self.assertEqual(mat.shape, (self.n_samples, EXTENDED_DIMENSION))
            self.assertEqual(mat.dtype, np.float32)
            self.assertTrue(np.allclose(mat, 1.0))

    def test_unverifiable_cache_missing_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for missing_key in ["manifest_sha256", "extractor_sha256", "settings_hash"]:
                cache_path = Path(tmp_dir) / f"missing_{missing_key}.npz"
                payload = {
                    "features": self.valid_features,
                    "image_ids": self.image_ids,
                    "split": self.split_name,
                    "manifest_sha256": self.manifest_sha,
                    "extractor_sha256": self.extractor_sha,
                    "settings_hash": self.settings_hash,
                }
                del payload[missing_key]
                np.savez_compressed(cache_path, **payload)

                with self.assertRaises(ValueError) as ctx:
                    validate_feature_cache(
                        cache_path, self.split_name, self.root, self.manifest
                    )
                self.assertIn("Unverifiable feature cache", str(ctx.exception))
                self.assertIn(missing_key, str(ctx.exception))

    def test_stale_manifest_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = self._save_cache(
                Path(tmp_dir), manifest_sha256="stale_manifest_hash_12345"
            )
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("manifest checksum mismatch", str(ctx.exception))

    def test_stale_extractor_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = self._save_cache(
                Path(tmp_dir), extractor_sha256="stale_extractor_hash_12345"
            )
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("extractor checksum mismatch", str(ctx.exception))

    def test_stale_settings_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = self._save_cache(
                Path(tmp_dir), settings_hash="stale_settings_hash_12345"
            )
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("feature extraction settings hash mismatch", str(ctx.exception))

    def test_split_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = self._save_cache(Path(tmp_dir), split="train")
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Cache split mismatch", str(ctx.exception))

    def test_image_ids_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrong_ids = np.array([f"wrong_{i:03d}.jpg" for i in range(self.n_samples)])
            cache_path = self._save_cache(Path(tmp_dir), image_ids=wrong_ids)
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Cache image IDs do not match", str(ctx.exception))

    def test_non_float32_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # float64
            f64 = self.valid_features.astype(np.float64)
            cache_path = self._save_cache(Path(tmp_dir), features=f64)
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Invalid feature matrix dtype: expected float32", str(ctx.exception))

    def test_wrong_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1,806 baseline features instead of 2,760
            bad_shape = np.ones((self.n_samples, 1806), dtype=np.float32)
            cache_path = self._save_cache(Path(tmp_dir), features=bad_shape)
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Invalid feature matrix shape", str(ctx.exception))

    def test_non_finite_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nan_feats = self.valid_features.copy()
            nan_feats[0, 5] = np.nan
            cache_path = self._save_cache(Path(tmp_dir), features=nan_feats)
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Feature cache contains non-finite values", str(ctx.exception))

    def test_negative_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            neg_feats = self.valid_features.copy()
            neg_feats[1, 10] = -0.5
            cache_path = self._save_cache(Path(tmp_dir), features=neg_feats)
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    cache_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Feature cache contains negative values", str(ctx.exception))

    def test_malformed_cache_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "corrupt.npz"
            corrupt_path.write_bytes(b"not an npz archive content")
            with self.assertRaises(ValueError) as ctx:
                validate_feature_cache(
                    corrupt_path, self.split_name, self.root, self.manifest
                )
            self.assertIn("Malformed cache file", str(ctx.exception))

    def test_missing_cache_file_raises_not_found(self):
        missing = Path("/nonexistent/path/cache.npz")
        with self.assertRaises(FileNotFoundError):
            validate_feature_cache(missing, self.split_name, self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()

