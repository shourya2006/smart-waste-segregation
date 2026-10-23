"""Regression checks for opt-in descriptors and isolation of the training study."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting.extended_features import (
    COLOR_DIM,
    DIMENSION,
    REGION_DIM,
    extended_columns,
    extract_extended_features,
)
from waste_sorting.extended_study import derived_data, extract_training_matrix
from waste_sorting.models.svm import make_pipeline
from waste_sorting.training import TrainingData


class ExtendedStudyTests(unittest.TestCase):
    def test_histogram_contract_and_degenerate_inputs(self):
        for mode, color in [("RGB", "white"), ("RGB", "black"), ("RGBA", (0, 0, 0, 0)), ("L", 128)]:
            image = Image.new(mode, (64, 48), color)
            x = extract_extended_features(image)
            self.assertEqual(x.shape, (DIMENSION,))
            self.assertEqual(x.dtype, np.float32)
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue((x >= 0).all())
            np.testing.assert_array_equal(x, extract_extended_features(image))
            for region in range(6):
                start = region * REGION_DIM
                for block in [
                    x[start : start + COLOR_DIM],
                    x[start + COLOR_DIM : start + REGION_DIM],
                ]:
                    self.assertTrue(np.isclose(block.sum(), 1) or block.sum() == 0)

    def test_selected_pipeline_persistence(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(size=(12, DIMENSION)).astype(np.float32)
        y = np.tile(np.arange(6), 2)
        for family in [
            "global_color",
            "global_color_texture",
            "global_color_spatial_texture",
            "spatial_color_texture",
        ]:
            columns = extended_columns(family)
            self.assertEqual(columns, sorted(set(columns)))
            model = make_pipeline("rbf_svm", columns, normalization="root_l2").fit(X, y)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.joblib"
                joblib.dump(model, path)
                np.testing.assert_array_equal(model.predict(X), joblib.load(path).predict(X))
        with self.assertRaises(ValueError):
            extended_columns("invalid")

    def test_identity_and_new_feature_fingerprint(self):
        base = TrainingData(
            np.ones((2, 4), dtype=np.float32),
            np.array([0, 1]),
            np.array(["a", "b"]),
            np.array(["g1", "g2"]),
            ("a", "b"),
            (),
            {},
            {"training_features_sha256": "old"},
            "base",
        )
        matrix = np.ones((2, DIMENSION), dtype=np.float32)
        derived = derived_data(base, matrix)
        self.assertIs(derived.image_ids, base.image_ids)
        self.assertIs(derived.y, base.y)
        self.assertIs(derived.groups, base.groups)
        self.assertNotEqual(derived.fingerprint, base.fingerprint)
        self.assertFalse(derived.X.flags.writeable)
        with self.assertRaises(ValueError):
            derived_data(base, np.zeros((1, DIMENSION), dtype=np.float32))

    def test_holdout_image_is_rejected_before_read(self):
        base = TrainingData(
            np.ones((1, 4)),
            np.array([0]),
            np.array(["heldout.jpg"]),
            np.array(["g"]),
            ("a",),
            (),
            {},
            {},
            "base",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs/dataset").mkdir(parents=True)
            pd.DataFrame(
                [dict(image_id="heldout.jpg", split="test", included=True, sha256="unused")]
            ).to_csv(root / "docs/dataset/manifest.csv", index=False)
            with patch("waste_sorting.extended_study.Image.open") as opened:
                with self.assertRaisesRegex(ValueError, "training rows only"):
                    extract_training_matrix(base, root)
                opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
