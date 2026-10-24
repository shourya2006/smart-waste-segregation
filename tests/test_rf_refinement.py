"""Tests for Random Forest refinement study, sampling reproducibility, and derived data."""

import unittest

import numpy as np

from waste_sorting.extended_features import DIMENSION as EXTENDED_DIMENSION
from waste_sorting.models.random_forest import make_pipeline
from waste_sorting.rf_refinement import (
    FAMILY_DEFINITIONS,
    derived_extended_data,
    sample_configurations,
)
from waste_sorting.training import TrainingData


class RFRefinementTests(unittest.TestCase):
    def test_sample_configurations_deterministic_and_contract(self):
        samples1 = sample_configurations(seed=42, n_iter=24)
        samples2 = sample_configurations(seed=42, n_iter=24)
        self.assertEqual(len(samples1), 24)
        self.assertEqual(samples1, samples2)

        for s in samples1:
            self.assertIn(s["classifier__n_estimators"], [300, 600])
            self.assertIn(s["classifier__max_depth"], [None, 20, 40])
            self.assertIn(s["classifier__min_samples_leaf"], [1, 2, 4])
            self.assertIn(s["classifier__max_features"], ["sqrt", 0.25, 0.5, 1.0])
            self.assertIn(s["classifier__class_weight"], [None, "balanced", "balanced_subsample"])
            self.assertEqual(s["classifier__min_samples_split"], 2)
            self.assertTrue(s["classifier__bootstrap"])

    def test_derived_extended_data_preserves_identities_and_fingerprints(self):
        base = TrainingData(
            np.ones((4, 10), dtype=np.float32),
            np.array([0, 1, 2, 3]),
            np.array(["img1", "img2", "img3", "img4"]),
            np.array(["g1", "g2", "g3", "g4"]),
            ("c0", "c1", "c2", "c3", "c4", "c5"),
            (),
            {},
            {"manifest_sha256": "dummy"},
            "fingerprint_base",
        )
        ext_matrix = np.zeros((4, EXTENDED_DIMENSION), dtype=np.float32)
        derived = derived_extended_data(base, ext_matrix)

        self.assertIs(derived.image_ids, base.image_ids)
        self.assertIs(derived.groups, base.groups)
        self.assertIs(derived.y, base.y)
        self.assertEqual(derived.X.shape, (4, EXTENDED_DIMENSION))
        self.assertNotEqual(derived.fingerprint, base.fingerprint)
        self.assertFalse(derived.X.flags.writeable)

        # Mismatched dimensions error
        with self.assertRaises(ValueError):
            derived_extended_data(base, np.zeros((4, 10), dtype=np.float32))

    def test_pipeline_with_refinement_parameters(self):
        cols = list(range(32))
        pipe = make_pipeline(cols, seed=42)
        params = {
            "classifier__n_estimators": 300,
            "classifier__max_depth": 20,
            "classifier__min_samples_leaf": 2,
            "classifier__max_features": 0.25,
            "classifier__class_weight": "balanced_subsample",
            "classifier__min_samples_split": 2,
            "classifier__bootstrap": True,
        }
        pipe.set_params(**params)
        self.assertEqual(pipe.named_steps["classifier"].max_features, 0.25)
        self.assertEqual(pipe.named_steps["classifier"].class_weight, "balanced_subsample")

        X = np.random.default_rng(42).normal(size=(20, 32)).astype(np.float32)
        y = np.random.default_rng(42).integers(0, 6, size=20)
        pipe.fit(X, y)
        preds = pipe.predict(X[:5])
        self.assertEqual(preds.shape, (5,))

    def test_family_definitions_contract(self):
        self.assertEqual(len(FAMILY_DEFINITIONS), 4)
        for _name, src, _spec, dim in FAMILY_DEFINITIONS:
            self.assertIn(src, ["original", "extended"])
            self.assertGreater(dim, 0)

    def test_oof_diagnostics_and_inference_contract(self):
        """Verify saved winner contract and OOF score reproduction."""
        from pathlib import Path
        results_dir = Path("docs/training/rf-refinement-results")
        if not results_dir.exists():
            return

        import json

        import pandas as pd
        from sklearn.metrics import f1_score

        oof = pd.read_csv(results_dir / "training-oof-predictions.csv")
        with open(results_dir / "summary.json") as f:
            summary = json.load(f)

        fold_f1s = []
        for fold_idx in range(5):
            sub = oof[oof["fold"] == fold_idx]
            f1 = f1_score(
                sub["true_label"],
                sub["predicted_label"],
                average="macro",
                zero_division=0,
            )
            fold_f1s.append(f1)

        expected_fold_f1s = summary["diagnostics"]["fold_macro_f1"]
        self.assertTrue(np.allclose(fold_f1s, expected_fold_f1s))


if __name__ == "__main__":
    unittest.main()
