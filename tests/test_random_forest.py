"""Tests for Random Forest pipeline, feature selection, training CV, and CLI."""

import contextlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from waste_sorting.acquisition import sha256
from waste_sorting.features import (
    compute_config_hash,
    get_descriptor_boundaries,
    get_feature_names,
    get_source_fingerprint,
)
from waste_sorting.models.random_forest import make_pipeline, parameter_grid
from waste_sorting.training import (
    feature_columns,
    load_or_create_folds,
    load_training_data,
    main,
    run_experiment,
)

REPO = Path(__file__).resolve().parents[1]


class RandomForestTests(unittest.TestCase):
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
        np.savez_compressed(self.cache, **self.payload)

    def load(self):
        return load_training_data(self.cache, self.root)

    def test_pipeline_column_selection_and_no_scaler(self):
        data = self.load()
        # Test individual descriptor dimensions
        color_cols = feature_columns(data, ["color"])
        hog_cols = feature_columns(data, ["hog"])
        lbp_cols = feature_columns(data, ["lbp"])
        color_hog_cols = feature_columns(data, ["color", "hog"])
        all_cols = feature_columns(data, ["color", "hog", "lbp"])

        self.assertEqual(len(color_cols), 32)
        self.assertEqual(len(hog_cols), 1764)
        self.assertEqual(len(lbp_cols), 10)
        self.assertEqual(len(color_hog_cols), 1796)
        self.assertEqual(len(all_cols), 1806)

        pipeline = make_pipeline(color_cols, seed=42)
        self.assertNotIn("scale", pipeline.named_steps)
        self.assertIn("features", pipeline.named_steps)
        self.assertIn("classifier", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["classifier"], RandomForestClassifier)

        # Empty columns rejected
        with self.assertRaises(ValueError):
            make_pipeline([])

    def test_parameter_grid_parses_configuration(self):
        grid = parameter_grid(self.config)
        self.assertIn("classifier__n_estimators", grid)
        self.assertIn("classifier__max_depth", grid)
        self.assertIn("classifier__min_samples_split", grid)
        self.assertIn("classifier__class_weight", grid)
        self.assertEqual(grid["classifier__class_weight"], [None, "balanced"])
        self.assertEqual(grid["classifier__n_estimators"], [100, 200])
        self.assertEqual(grid["classifier__max_depth"], [None, 10, 20])
        self.assertEqual(grid["classifier__min_samples_split"], [2, 5])

        # Missing section error
        bad_config = {"training": {}}
        with self.assertRaises(ValueError):
            parameter_grid(bad_config)

    def test_rf_experiment_execution_reproducibility_and_saved_model(self):
        data = self.load()
        folds = load_or_create_folds(data, self.fold_path)
        descriptors = ["color"]
        columns = feature_columns(data, descriptors)
        pipeline = make_pipeline(columns, seed=42)

        small_grid = {
            "classifier__n_estimators": [10],
            "classifier__max_depth": [5],
            "classifier__class_weight": [None, "balanced"],
        }
        out_dir = self.root / "rf_run"
        summary = run_experiment(
            data,
            folds,
            pipeline,
            small_grid,
            model_name="random_forest",
            descriptors=descriptors,
            output_dir=out_dir,
            root=self.root,
            fold_path=self.fold_path,
            seed=42,
        )

        self.assertEqual(summary["model"], "random_forest")
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(len(summary["fold_macro_f1"]), 5)
        self.assertEqual(summary["selected_feature_dimension"], 32)
        self.assertEqual(summary["input_feature_dimension"], 1806)
        self.assertFalse(summary["calibrated"])
        self.assertFalse(summary["held_out_predictions_generated"])

        # Reload saved model and test prediction with full feature vector
        saved_model = joblib.load(out_dir / "estimator.joblib")
        # Ensure it accepts full 1806-dim input and predicts classes
        preds = saved_model.predict(data.X[:5])
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(all(0 <= p < 6 for p in preds))

        # Check cv_results.csv was generated
        results_df = pd.read_csv(out_dir / "cv_results.csv")
        self.assertEqual(len(results_df), 2)
        self.assertIn("mean_test_score", results_df.columns)

    def test_cli_rf_subcommand(self):
        out_path = self.root / "cli_rf_run"

        # Replace [training.random_forest] section in config_text to keep fast grid
        base_text = self.config_text
        if "[training.random_forest]" in base_text:
            base_text = base_text.split("[training.random_forest]")[0]
        lines = [
            base_text,
            "[training.random_forest]",
            "n_estimators = [5]",
            "max_depth = [3]",
            "min_samples_split = [2]",
            "class_weight = ['none']",
        ]
        fast_config_path = self.root / "configs/fast.toml"
        fast_config_path.write_text("\n".join(lines) + "\n")

        cli_args = [
            "--root",
            str(self.root),
            "--config",
            "configs/fast.toml",
            "--cache",
            "features.npz",
            "--folds",
            "folds.json",
            "rf",
            "--feature-sets",
            "color",
            "--output",
            str(out_path),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            main(cli_args)

        run_dir = out_path / "random_forest_color"
        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "estimator.joblib").exists())
        self.assertTrue((run_dir / "cv_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
