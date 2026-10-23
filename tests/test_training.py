"""Checks for row identity, group isolation, frozen folds and fold-local learning."""

import contextlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from waste_sorting.acquisition import sha256
from waste_sorting.features import (
    compute_config_hash,
    get_descriptor_boundaries,
    get_feature_names,
    get_source_fingerprint,
)
from waste_sorting.models.svm import make_pipeline, parameter_grid
from waste_sorting.training import (
    feature_columns,
    load_or_create_folds,
    load_training_data,
    main,
    make_folds,
    run_experiment,
)

REPO = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "docs/dataset").mkdir(parents=True)
        (self.root / "requirements.lock").write_text("synthetic test fixture\n")
        self.config_text = (REPO / "configs/phase1.toml").read_text()
        (self.root / "configs/phase1.toml").write_text(self.config_text)
        self.config = tomllib.loads(self.config_text)
        classes = self.config["classes"]
        rows = []
        # Two related images in each training group; outer held-out values are extreme
        # so accidental inclusion would be visible in fitted scaling statistics.
        for label in classes:
            for group in range(20):
                for copy in range(2):
                    rows.append(
                        {
                            "image_id": f"{label}/train-{group:02d}-{copy}.jpg",
                            "label": label,
                            "included": True,
                            "split": "train",
                            "duplicate_group": f"{label}-{group}",
                        }
                    )
            for split in ("validation", "test"):
                rows.append(
                    {
                        "image_id": f"{label}/{split}.jpg",
                        "label": label,
                        "included": True,
                        "split": split,
                        "duplicate_group": f"{label}-{split}",
                    }
                )
        rows.append(
            {
                "image_id": "trash/excluded.jpg",
                "label": "trash",
                "included": False,
                "split": "excluded",
                "duplicate_group": "excluded",
            }
        )
        self.manifest = pd.DataFrame(rows)
        self.manifest_path = self.root / "docs/dataset/manifest.csv"
        self.manifest.to_csv(self.manifest_path, index=False)
        included = self.manifest[self.manifest["included"]].sort_values("image_id")
        rng = np.random.default_rng(42)
        X = rng.normal(size=(len(included), 1806)).astype(np.float32)
        X[included["split"].to_numpy() != "train"] = 1e6
        metadata = {
            "manifest_sha256": sha256(self.manifest_path),
            "config_hash": compute_config_hash(self.config),
            "source_fingerprint": get_source_fingerprint(),
            "sample_count": len(included),
            "feature_dim": 1806,
            "descriptor_boundaries": get_descriptor_boundaries(self.config),
        }
        self.payload = {
            "features": X,
            "image_ids": included["image_id"].to_numpy(dtype=str),
            "labels": included["label"].to_numpy(dtype=str),
            "targets": np.array([classes.index(label) for label in included["label"]]),
            "splits": included["split"].to_numpy(dtype=str),
            "duplicate_groups": included["duplicate_group"].to_numpy(dtype=str),
            "relative_paths": included["image_id"].to_numpy(dtype=str),
            "classes": np.array(classes),
            "feature_names": np.array(get_feature_names(self.config)),
            "metadata": np.array(json.dumps(metadata)),
        }
        self.cache = self.root / "features.npz"
        self.fold_path = self.root / "folds.json"
        self.write_cache()

    def write_cache(self):
        np.savez_compressed(self.cache, **self.payload)

    def load(self):
        return load_training_data(self.cache, self.root)

    def test_loader_excludes_held_out_rows_and_preserves_identity(self):
        data = self.load()
        expected = self.manifest[self.manifest["split"] == "train"].sort_values("image_id")
        self.assertEqual(data.X.shape, (240, 1806))
        np.testing.assert_array_equal(data.image_ids, expected["image_id"])
        np.testing.assert_array_equal(data.groups, expected["duplicate_group"])
        self.assertLess(float(data.X.max()), 10)
        self.assertFalse(data.X.flags.writeable)

    def test_misaligned_id_label_group_split_or_target_is_rejected(self):
        for key in ("image_ids", "labels", "duplicate_groups", "splits", "targets"):
            with self.subTest(key=key):
                original = self.payload[key].copy()
                self.payload[key] = np.roll(original, 1)
                self.write_cache()
                with self.assertRaisesRegex(ValueError, "Cached"):
                    self.load()
                self.payload[key] = original

    def test_excluded_rows_cannot_be_appended_to_training_cache(self):
        self.payload["image_ids"] = np.append(self.payload["image_ids"], "trash/excluded.jpg")
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "image_ids"):
            self.load()

    def test_nonfinite_or_wrong_shape_feature_matrix_is_rejected(self):
        original = self.payload["features"].copy()
        self.payload["features"][0, 0] = np.nan
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.load()
        self.payload["features"] = original[:, :-1]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "float32 matrix"):
            self.load()

    def test_outer_split_group_leakage_is_rejected(self):
        self.manifest.loc[self.manifest["image_id"] == "glass/test.jpg", "duplicate_group"] = (
            "glass-0"
        )
        self.manifest.to_csv(self.manifest_path, index=False)
        metadata = json.loads(str(self.payload["metadata"]))
        metadata["manifest_sha256"] = sha256(self.manifest_path)
        self.payload["metadata"] = np.array(json.dumps(metadata))
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "crosses outer split"):
            self.load()

    def test_fold_assignments_are_reproducible_exhaustive_and_group_isolated(self):
        data = self.load()
        folds = load_or_create_folds(data, self.fold_path)
        before = self.fold_path.read_bytes()
        again = load_or_create_folds(data, self.fold_path)
        self.assertEqual(before, self.fold_path.read_bytes())
        visits = np.zeros(len(data.y))
        for (train, valid), (train2, valid2) in zip(folds, again, strict=True):
            self.assertFalse(set(data.groups[train]) & set(data.groups[valid]))
            self.assertEqual(set(data.y[valid]), set(range(6)))
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(valid, valid2)
            visits[valid] += 1
        np.testing.assert_array_equal(visits, np.ones(len(data.y)))

    def test_saved_fold_edits_seed_changes_and_feature_changes_are_rejected(self):
        data = self.load()
        load_or_create_folds(data, self.fold_path)
        with self.assertRaisesRegex(ValueError, "Saved folds differ"):
            load_or_create_folds(data, self.fold_path, seed=43)
        training_row = np.flatnonzero(self.payload["splits"] == "train")[0]
        self.payload["features"][training_row, 0] += 1
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "Saved folds differ"):
            load_or_create_folds(self.load(), self.fold_path)
        payload = json.loads(self.fold_path.read_text())
        payload["assignments"][str(data.image_ids[0])] = 99
        self.fold_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Saved folds differ"):
            load_or_create_folds(data, self.fold_path)

    def test_insufficient_group_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct training groups"):
            make_folds(self.load(), n_splits=21)

    def test_scaler_fits_only_fold_training_rows_and_final_outer_train(self):
        data = self.load()
        folds = load_or_create_folds(data, self.fold_path)
        columns = feature_columns(data, ["color"])
        estimator = make_pipeline("linear_svm", columns)
        observed = []
        real_fit = StandardScaler.fit

        def record_fit(scaler, X, y=None, **kwargs):
            observed.append(np.asarray(X).mean(axis=0))
            return real_fit(scaler, X, y, **kwargs)

        with patch.object(StandardScaler, "fit", record_fit):
            summary = run_experiment(
                data,
                folds,
                estimator,
                {"classifier__C": [0.1]},
                model_name="linear_svm",
                descriptors=["color"],
                output_dir=self.root / "linear",
                root=self.root,
                fold_path=self.fold_path,
            )
        self.assertEqual(len(observed), 6)
        for mean, (train, _) in zip(observed[:5], folds, strict=True):
            np.testing.assert_allclose(mean, data.X[train][:, columns].mean(axis=0))
        np.testing.assert_allclose(observed[-1], data.X[:, columns].mean(axis=0))
        saved = joblib.load(self.root / "linear/estimator.joblib")
        self.assertEqual(saved.predict(data.X[:2]).shape, (2,))
        self.assertFalse(summary["held_out_predictions_generated"])
        self.assertFalse(summary["calibrated"])
        self.assertEqual(summary["fold_file_sha256"], sha256(self.fold_path))
        self.assertAlmostEqual(summary["mean_cv_macro_f1"], np.mean(summary["fold_macro_f1"]))

    def test_experiment_cannot_use_folds_different_from_logged_protocol(self):
        data = self.load()
        original = load_or_create_folds(data, self.fold_path)
        with self.assertRaisesRegex(ValueError, "do not match the shared fold"):
            run_experiment(
                data,
                list(reversed(original)),
                make_pipeline("dummy", feature_columns(data)),
                {},
                model_name="dummy",
                descriptors=["color", "hog", "lbp"],
                output_dir=self.root / "wrong-folds",
                root=self.root,
                fold_path=self.fold_path,
            )

    def test_svm_pipeline_selection_and_predeclared_searches(self):
        data = self.load()
        self.assertEqual(len(feature_columns(data, ["hog", "color"])), 1796)
        for name in ("linear_svm", "rbf_svm"):
            pipeline = make_pipeline(name, feature_columns(data))
            self.assertIsInstance(pipeline.named_steps["scale"], StandardScaler)
            grid = parameter_grid(name, self.config)
            self.assertEqual(grid["classifier__class_weight"], [None, "balanced"])
        with self.assertRaises(ValueError):
            feature_columns(data, ["cnn"])

    def test_cli_creates_folds_and_dummy_artifact_without_overwriting(self):
        arguments = ["--root", str(self.root), "--cache", "features.npz", "--folds", "folds.json"]
        with contextlib.redirect_stdout(io.StringIO()):
            main([*arguments, "folds"])
            main([*arguments, "svm", "--models", "dummy", "--output", "run"])
        summary = json.loads((self.root / "run/dummy/summary.json").read_text())
        self.assertEqual(summary["training_samples"], 240)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(len(summary["fold_macro_f1"]), 5)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            main([*arguments, "svm", "--models", "dummy", "--output", "run"])


if __name__ == "__main__":
    unittest.main()
