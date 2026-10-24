"""Regression checks for the saved-record audit; no experiment fitting."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.verify_rf_refinement import audit, comparison_rows

ROOT = Path(__file__).resolve().parents[1]


class RFReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["docs/training", "docs/dataset", "src/waste_sorting"]:
            shutil.copytree(ROOT / name, self.root / name)
        self.study = self.root / "docs/training/rf-refinement-results"

    def test_saved_records_and_correct_historical_accuracy_without_fitting(self):
        with patch(
            "sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("fit")
        ):
            summary, result = audit(self.root, self.study)
        self.assertEqual(result["candidates_verified"], 96)
        self.assertEqual(result["recorded_fold_scores_verified"], 480)
        self.assertEqual(result["correct_predictions"], 1490)
        self.assertFalse(result["fitting_performed"])
        rows = {r["model"]: r for r in comparison_rows(self.root, summary)}
        self.assertEqual(rows["Original HSV RF"]["pooled_oof_accuracy"], 0.753968253968254)
        self.assertEqual(rows["Compact RBF SVM"]["pooled_oof_accuracy"], 0.8197278911564626)
        for model in ["Dummy baseline", "Original linear SVM", "Original RBF SVM"]:
            self.assertIsNone(rows[model]["pooled_oof_accuracy"])

    def test_rejects_candidate_score_drift(self):
        path = self.study / "candidate-results.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "split2_test_score"] += 0.01
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Candidate mean"):
            audit(self.root, self.study)

    def test_rejects_different_configuration_for_same_index(self):
        path = self.study / "candidate-results.csv"
        rows = pd.read_csv(path)
        rows.loc[24, "params"] = rows.loc[25, "params"]
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Candidate configuration"):
            audit(self.root, self.study)

    def test_rejects_frozen_fold_assignment_drift(self):
        path = self.study / "training-oof-predictions.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "fold"] = (rows.loc[0, "fold"] + 1) % 5
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "frozen fold assignment"):
            audit(self.root, self.study)

    def test_rejects_historical_accuracy_not_supported_by_counts(self):
        path = self.root / "docs/training/svm-study-results/summary.json"
        data = json.loads(path.read_text())
        data["diagnostics"]["random_forest_color"]["accuracy"] = 0.7557
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Historical accuracy"):
            audit(self.root, self.study)

    def test_rejects_duplicate_or_replaced_training_identity(self):
        path = self.study / "training-oof-predictions.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "image_id"] = rows.loc[1, "image_id"]
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "each outer-training image"):
            audit(self.root, self.study)


if __name__ == "__main__":
    unittest.main()
