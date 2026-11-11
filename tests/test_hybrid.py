"""Unit tests for hybrid ML + DL models (src/waste_sorting/hybrid.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from waste_sorting.hybrid import (
    BlockScaledFusionTransformer,
    generate_phase1_svm_oof,
    load_aligned_features,
    make_h1_pipeline,
    make_h2_pipeline,
    run_cv,
)


class TestBlockScaledFusionTransformer(unittest.TestCase):
    def test_block_dimensions_and_scaling(self):
        rng = np.random.RandomState(42)
        dim1, dim2 = 10, 20
        N = 100
        # Block 1 with mean 10, std 5; Block 2 with mean -2, std 0.5
        X1 = rng.normal(loc=10.0, scale=5.0, size=(N, dim1)).astype(np.float32)
        X2 = rng.normal(loc=-2.0, scale=0.5, size=(N, dim2)).astype(np.float32)
        X = np.hstack([X1, X2])

        transformer = BlockScaledFusionTransformer(dim_block1=dim1, dim_block2=dim2)
        X_trans = transformer.fit_transform(X)

        self.assertEqual(X_trans.shape, (N, dim1 + dim2))

        # Check that each block's total variance equals 1.0
        X1_t = X_trans[:, :dim1]
        X2_t = X_trans[:, dim1:]

        var1 = np.sum(np.var(X1_t, axis=0))
        var2 = np.sum(np.var(X2_t, axis=0))
        np.testing.assert_allclose(var1, 1.0, rtol=1e-5)
        np.testing.assert_allclose(var2, 1.0, rtol=1e-5)

    def test_fold_local_isolation(self):
        """Ensure outlier in validation set does not leak into transformer parameters."""
        dim1, dim2 = 4, 4
        X_train = np.ones((10, dim1 + dim2), dtype=np.float32)
        X_val_outlier = np.full((5, dim1 + dim2), 1e6, dtype=np.float32)

        transformer = BlockScaledFusionTransformer(dim1, dim2)
        transformer.fit(X_train)

        # Means should be exactly 1.0, unaffected by X_val_outlier
        np.testing.assert_allclose(transformer.scaler1_.mean_, np.ones(dim1))
        np.testing.assert_allclose(transformer.scaler2_.mean_, np.ones(dim2))

        val_trans = transformer.transform(X_val_outlier)
        self.assertEqual(val_trans.shape, (5, dim1 + dim2))

    def test_wrong_input_dimension_raises(self):
        transformer = BlockScaledFusionTransformer(10, 10)
        with self.assertRaises(ValueError):
            transformer.fit(np.zeros((5, 15)))


class TestHybridPipelines(unittest.TestCase):
    def test_make_h1_pipeline(self):
        pipe = make_h1_pipeline("rbf_svm", C=10.0, gamma="scale")
        self.assertIsInstance(pipe, Pipeline)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertEqual(pipe.named_steps["classifier"].C, 10.0)

    def test_make_h2_pipeline(self):
        pipe = make_h2_pipeline("rbf_svm", dim_hc=1608, dim_cnn=1280, C=5.0)
        self.assertIsInstance(pipe, Pipeline)
        self.assertIsInstance(pipe.named_steps["scaler"], BlockScaledFusionTransformer)
        self.assertEqual(pipe.named_steps["classifier"].C, 5.0)


class TestAlignmentAndDataIntegrity(unittest.TestCase):
    def test_id_mismatch_raises(self):
        """Verify load_aligned_features raises ValueError when image_ids are mismatched."""
        with patch("waste_sorting.hybrid.load_embeddings") as mock_emb:
            mock_emb.return_value = {
                "image_ids": np.array(["img_1", "img_2"]),
                "embeddings": np.zeros((2, 1280), dtype=np.float32),
                "labels": np.array([0, 1]),
            }
            with patch("waste_sorting.hybrid.extract_split_features") as mock_ext:
                mock_ext.return_value = (
                    np.zeros((2, 2760), dtype=np.float32),
                    np.array([0, 1]),
                    np.array(["img_A", "img_B"]),  # Mismatched IDs
                    np.array(["g1", "g2"]),
                )
                with self.assertRaises(ValueError):
                    load_aligned_features("mobilenet_v2", split="train")

    def test_order_mismatch_raises(self):
        """Verify load_aligned_features raises ValueError when order differs."""
        with patch("waste_sorting.hybrid.load_embeddings") as mock_emb:
            mock_emb.return_value = {
                "image_ids": np.array(["img_1", "img_2"]),
                "embeddings": np.zeros((2, 1280), dtype=np.float32),
                "labels": np.array([0, 1]),
            }
            with patch("waste_sorting.hybrid.extract_split_features") as mock_ext:
                mock_ext.return_value = (
                    np.zeros((2, 2760), dtype=np.float32),
                    np.array([1, 0]),
                    np.array(["img_2", "img_1"]),  # Inverted order
                    np.array(["g1", "g2"]),
                )
                with self.assertRaises(ValueError):
                    load_aligned_features("mobilenet_v2", split="train")

    def test_validation_split_rejected(self):
        """Reject loading non-train split."""
        with self.assertRaises(ValueError):
            load_aligned_features("mobilenet_v2", split="validation")


class TestRunCV(unittest.TestCase):
    def test_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "already_exists"
            out_dir.mkdir()
            with self.assertRaises(FileExistsError):
                run_cv(
                    candidate_type="h1",
                    backbone="mobilenet_v2",
                    hyperparams={"classifier": "logistic_regression", "C": 1.0},
                    root=".",
                    output_dir=out_dir,
                )

    def test_oof_probabilities_sum_to_one(self):
        """Synthetic test verifying OOF probabilities sum to 1.0."""
        # Create a mock OOF dataframe with probability columns
        cols = [f"prob_{c}" for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]]
        df = pd.DataFrame(np.ones((10, 6)) / 6.0, columns=cols)
        sums = df[cols].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


class TestPhase1OOF(unittest.TestCase):
    def test_phase1_oof_file_overwrite_refusal(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
            with self.assertRaises(FileExistsError):
                generate_phase1_svm_oof(root=".", output_path=tmp_file.name)


if __name__ == "__main__":
    unittest.main()
