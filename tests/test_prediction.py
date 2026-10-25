"""Unit tests for the prediction interface and artifact validation."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from waste_sorting.evaluation import CLASSES
from waste_sorting.extended_features import DIMENSION as EXTENDED_DIMENSION
from waste_sorting.prediction import (
    CANONICAL_CLASSES,
    PredictionResult,
    predict_image,
    validate_artifact,
)


class DummyCalibratedModel:
    """Mock calibrated model returning deterministic probabilities for testing."""

    def __init__(self, probabilities=None):
        self.classes_ = np.arange(len(CLASSES))
        self.n_features_in_ = EXTENDED_DIMENSION
        self.probabilities = (
            probabilities
            if probabilities is not None
            else np.array([0.10, 0.05, 0.70, 0.05, 0.05, 0.05], dtype=np.float64)
        )

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


class PredictionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.dummy_model = DummyCalibratedModel()
        self.valid_artifact = {
            "calibrated_model": self.dummy_model,
            "classes": list(CLASSES),
            "threshold": 0.65,
            "input_dimension": EXTENDED_DIMENSION,
            "model_type": "calibrated_ensemble",
            "base_model": "extended_rbf_svm",
        }
        self.test_image = Image.new("RGB", (64, 64), color=(200, 100, 50))

    def test_predict_image_returns_valid_structure(self):
        res = predict_image(self.test_image, self.valid_artifact)
        self.assertIsInstance(res, PredictionResult)
        self.assertEqual(res.predicted_class, "metal")  # index 2 has 0.70 prob
        self.assertEqual(len(res.probabilities), 6)
        self.assertEqual(list(res.probabilities.keys()), list(CLASSES))
        self.assertAlmostEqual(res.max_probability, 0.70)
        self.assertFalse(res.abstained)  # 0.70 >= 0.65

        # Dict-like compatibility
        self.assertEqual(res["predicted_class"], "metal")
        self.assertAlmostEqual(res["max_probability"], 0.70)
        self.assertFalse(res["abstained"])
        self.assertIn("cardboard", res["probabilities"])

    def test_abstention_flag_when_max_prob_below_threshold(self):
        # Threshold higher than max probability (0.70)
        artifact_high_threshold = dict(self.valid_artifact)
        artifact_high_threshold["threshold"] = 0.75

        res = predict_image(self.test_image, artifact_high_threshold)
        self.assertTrue(res.abstained)
        self.assertAlmostEqual(res.max_probability, 0.70)

    def test_prediction_consistency_after_saving_and_reloading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_file = Path(tmp_dir) / "test_artifact.joblib"
            image_file = Path(tmp_dir) / "test_image.jpg"
            self.test_image.save(image_file)
            joblib.dump(self.valid_artifact, artifact_file)

            # Predict from in-memory dict and image
            res_mem = predict_image(self.test_image, self.valid_artifact)
            # Predict from saved file paths
            res_file = predict_image(image_file, artifact_file)

            self.assertEqual(res_mem.predicted_class, res_file.predicted_class)
            self.assertAlmostEqual(res_mem.max_probability, res_file.max_probability)
            self.assertEqual(res_mem.abstained, res_file.abstained)
            for k in CLASSES:
                self.assertAlmostEqual(res_mem.probabilities[k], res_file.probabilities[k])

    def test_artifact_validation_missing_keys(self):
        bad_artifact = dict(self.valid_artifact)
        del bad_artifact["threshold"]
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("missing required metadata keys", str(ctx.exception))

    def test_artifact_dimension_mismatch_rejects_1806_features(self):
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["input_dimension"] = 1806  # baseline dimension
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("1,806-feature baseline cache cannot be substituted", str(ctx.exception))

    def test_artifact_invalid_class_count(self):
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["classes"] = ["cardboard", "glass", "metal"]
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("must exactly match canonical order", str(ctx.exception))

    def test_artifact_invalid_threshold(self):
        for bad_thresh in [-0.1, 1.5, "high"]:
            bad_artifact = dict(self.valid_artifact)
            bad_artifact["threshold"] = bad_thresh
            with self.assertRaises(ValueError):
                validate_artifact(bad_artifact)

    def test_corrupt_or_missing_image_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_file = Path(tmp_dir) / "nonexistent.jpg"
            with self.assertRaises(FileNotFoundError):
                predict_image(missing_file, self.valid_artifact)

            corrupt_file = Path(tmp_dir) / "corrupt.jpg"
            corrupt_file.write_bytes(b"not an image file content")
            with self.assertRaises(ValueError) as ctx:
                predict_image(corrupt_file, self.valid_artifact)
            self.assertIn("Corrupt or unreadable image file", str(ctx.exception))

    def test_artifact_rejects_reversed_class_metadata(self):
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["classes"] = list(reversed(CANONICAL_CLASSES))
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("must exactly match canonical order", str(ctx.exception))

    def test_artifact_rejects_permuted_class_metadata(self):
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["classes"] = [
            "glass",
            "cardboard",
            "metal",
            "paper",
            "plastic",
            "trash",
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("must exactly match canonical order", str(ctx.exception))

    def test_artifact_rejects_missing_model_class_order(self):
        del self.dummy_model.classes_
        with self.assertRaisesRegex(ValueError, "missing classes_"):
            validate_artifact(self.valid_artifact)

    def test_artifact_rejects_invalid_model_class_metadata(self):
        for invalid in [None, 6, "cardboard", np.array([[0, 1, 2, 3, 4, 5]])]:
            with self.subTest(classes=invalid):
                self.dummy_model.classes_ = invalid
                with self.assertRaisesRegex(ValueError, "classes_"):
                    validate_artifact(self.valid_artifact)

    def test_artifact_rejects_mismatched_model_class_order(self):
        bad_model = DummyCalibratedModel()
        bad_model.classes_ = np.array([5, 4, 3, 2, 1, 0])  # reversed class order
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["calibrated_model"] = bad_model
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("does not match expected", str(ctx.exception))

    def test_artifact_rejects_model_input_dimension_mismatch(self):
        bad_model = DummyCalibratedModel()
        bad_model.n_features_in_ = 1806  # baseline dimension instead of 2760
        bad_artifact = dict(self.valid_artifact)
        bad_artifact["calibrated_model"] = bad_model
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("Model input dimension mismatch", str(ctx.exception))

    def test_artifact_rejects_calibrated_fold_dimension_mismatch(self):
        class MockClassifier:
            def __init__(self, estimator):
                self.estimator = estimator

        class MockBaseEstimator:
            def __init__(self, n_features, classes):
                self.n_features_in_ = n_features
                self.classes_ = classes

        bad_model = DummyCalibratedModel()
        bad_base = MockBaseEstimator(1806, np.arange(6))
        bad_model.calibrated_classifiers_ = [MockClassifier(bad_base)]

        bad_artifact = dict(self.valid_artifact)
        bad_artifact["calibrated_model"] = bad_model
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("base estimator input dimension mismatch", str(ctx.exception))

    def test_artifact_rejects_calibrated_fold_class_order_mismatch(self):
        class MockClassifier:
            def __init__(self, estimator):
                self.estimator = estimator

        class MockBaseEstimator:
            def __init__(self, n_features, classes):
                self.n_features_in_ = n_features
                self.classes_ = classes

        bad_model = DummyCalibratedModel()
        bad_base = MockBaseEstimator(EXTENDED_DIMENSION, np.array([1, 0, 2, 3, 4, 5]))
        bad_model.calibrated_classifiers_ = [MockClassifier(bad_base)]

        bad_artifact = dict(self.valid_artifact)
        bad_artifact["calibrated_model"] = bad_model
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(bad_artifact)
        self.assertIn("base estimator classes_ [1 0 2 3 4 5] does not match", str(ctx.exception))

    def test_invalid_image_type_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            predict_image(12345, self.valid_artifact)
        self.assertIn("Expected PIL Image or image path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

