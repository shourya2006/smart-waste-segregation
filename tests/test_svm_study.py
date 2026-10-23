"""Selection, normalization, parallel convergence, OOF scope and study replay."""

import contextlib
import io
import json
import unittest

import joblib
import numpy as np
import pandas as pd
import test_training
from sklearn.exceptions import ConvergenceWarning

from waste_sorting.models.svm import StrictLinearSVC, make_pipeline, sqrt_features
from waste_sorting.svm_study import (
    refinement_grid,
    refinement_winners,
    run_study,
    training_diagnostics,
)
from waste_sorting.training import feature_columns, load_or_create_folds


class SVMStudyTests(unittest.TestCase):
    def fixture(self):
        case = test_training.TrainingTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        # Real cached descriptors are nonnegative. Keep held-out extremes in the
        # fixture to catch accidental outer-validation/test use.
        case.payload["features"] = np.abs(case.payload["features"])
        case.write_cache()
        return case

    def test_refinement_uses_only_top_rbf_families_with_stable_ties(self):
        runs = [
            {"model": "linear_svm", "feature_set": "hog", "mean_cv_macro_f1": 0.99},
            {"model": "rbf_svm", "feature_set": "lbp", "mean_cv_macro_f1": 0.5},
            {"model": "rbf_svm", "feature_set": "color", "mean_cv_macro_f1": 0.8},
            {"model": "rbf_svm", "feature_set": "color_lbp", "mean_cv_macro_f1": 0.8},
        ]
        self.assertEqual(
            [r["feature_set"] for r in refinement_winners(runs, 2)], ["color", "color_lbp"]
        )
        with self.assertRaises(ValueError):
            refinement_winners(runs, 0)

    def test_root_normalization_is_per_row_and_handles_zero_vectors(self):
        pipeline = make_pipeline("rbf_svm", [0, 1], normalization="root_l2")
        scale = pipeline.named_steps["scale"]
        X = np.array([[1.0, 3.0], [0.0, 0.0]])
        transformed = scale.fit_transform(X)
        np.testing.assert_allclose(transformed, [[0.5, np.sqrt(3) / 2], [0.0, 0.0]])
        np.testing.assert_allclose(scale.transform(X[:1]), transformed[:1])
        for invalid in [np.array([[-1.0]]), np.array([[np.nan]])]:
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                sqrt_features(invalid)

    def test_refinement_gamma_depends_on_dimension_not_holdout_statistics(self):
        settings = {
            "refined_C": [1.0, 30.0],
            "class_weight": ["none", "balanced"],
            "standard_gamma_dimension_multipliers": [0.25, 4.0],
            "root_l2_gamma": ["scale", 1.0, 10.0],
        }
        self.assertEqual(
            refinement_grid(settings, 32, "standard")["classifier__gamma"],
            ["scale", 0.25 / 32, 4 / 32],
        )
        self.assertEqual(
            refinement_grid(settings, 32, "root_l2")["classifier__gamma"], ["scale", 1.0, 10.0]
        )

    def test_unconverged_linear_fits_raise_inside_parallel_workers(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 32))
        y = np.arange(120) % 6
        with self.assertRaises(ConvergenceWarning):
            joblib.Parallel(n_jobs=2)(
                joblib.delayed(StrictLinearSVC(C=100, max_iter=1, random_state=42).fit)(X, y)
                for _ in range(2)
            )

    def test_oof_diagnostics_contain_only_training_ids_and_correct_support(self):
        case = self.fixture()
        data = case.load()
        cv = load_or_create_folds(data, case.fold_path)
        estimator = make_pipeline("linear_svm", feature_columns(data, ["color"]))
        report = training_diagnostics(data, cv, estimator, case.root / "oof")
        predictions = pd.read_csv(case.root / "oof/training-oof-predictions.csv")
        np.testing.assert_array_equal(predictions.image_id, data.image_ids)
        self.assertEqual(np.array(report["confusion_matrix"]).sum(), 240)
        self.assertEqual(sum(r["support"] for r in report["per_class"].values()), 240)
        self.assertTrue(predictions["fold"].isin(range(5)).all())
        self.assertFalse(report["outer_holdout_predictions_generated"])
        self.assertAlmostEqual(report["mean_fold_macro_f1"], np.mean(report["fold_macro_f1"]))

    def test_small_two_stage_study_replays_without_refitting_and_rejects_changes(self):
        case = self.fixture()
        config = case.config_text.split("[svm_study]")[0]
        config = config.replace("C = [0.1, 1.0, 10.0]", "C = [0.1]")
        config = config.replace('class_weight = ["none", "balanced"]', 'class_weight = ["none"]')
        config = config.replace('gamma = ["scale", 0.001, 0.01]', 'gamma = ["scale"]')
        config += """
[svm_study]
feature_sets = ["color"]
linear_max_iter = 20000
refinement_top_k = 1
refined_C = [1.0]
standard_gamma_dimension_multipliers = []
root_l2_gamma = ["scale"]
class_weight = ["none"]
"""
        config_path = case.root / "configs/study.toml"
        config_path.write_text(config)
        output = case.root / "study"
        args = [case.root, case.cache, config_path, case.fold_path, output]
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_study(*args)
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(result["cv_fits"], 20)
        self.assertEqual(result["refined_feature_sets"], ["color"])
        self.assertEqual(len(pd.read_csv(output / "candidate-results.csv")), 4)
        for run in result["runs"]:
            self.assertEqual(run["training_samples"], 240)
        summary_before = json.loads((output / "study-summary.json").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run_study(*args, resume=True)
        self.assertEqual(summary_before, resumed)
        config_path.write_text(config.replace("refined_C = [1.0]", "refined_C = [10.0]"))
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            run_study(*args, resume=True)


if __name__ == "__main__":
    unittest.main()
