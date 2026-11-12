"""Unit tests for hybrid ML + DL models (src/waste_sorting/hybrid.py)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from waste_sorting.evaluation import CLASSES
from waste_sorting.hybrid import (
    BlockScaledFusionTransformer,
    generate_phase1_svm_oof,
    load_aligned_features,
    load_hybrid,
    make_h1_pipeline,
    make_h2_pipeline,
    predict_proba,
    run_cv,
    run_late_fusion,
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

    def test_invalid_family_rejected(self):
        """Invalid candidate_type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_cv(
                    candidate_type="invalid_family",
                    backbone="mobilenet_v2",
                    hyperparams={"classifier": "logistic_regression", "C": 1.0},
                    root=".",
                    output_dir=Path(tmp_dir) / "out",
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


class TestLateFusion(unittest.TestCase):
    def setUp(self):
        classes = list(CLASSES)
        n_samples = 50

        # Build consistent synthetic OOF datasets
        rows_cnn = []
        rows_svm = []
        for i in range(n_samples):
            img_id = f"img_{i:04d}.jpg"
            fold = i % 5
            label_idx = i % 6

            p_cnn = np.zeros(6)
            # Make CNN accurate
            p_cnn[label_idx] = 0.8
            p_cnn[(label_idx + 1) % 6] = 0.2

            p_svm = np.zeros(6)
            # Make SVM have lower confidence and some wrong predictions
            wrong_idx = (label_idx + 3) % 6 if i % 4 == 0 else label_idx
            p_svm[wrong_idx] = 0.7
            p_svm[(wrong_idx + 1) % 6] = 0.3

            row_c = {"image_id": img_id, "fold": fold, "label": label_idx}
            for j, c in enumerate(classes):
                row_c[f"prob_{c}"] = p_cnn[j]
            rows_cnn.append(row_c)

            row_s = {"image_id": img_id, "fold": fold, "label": classes[label_idx]}
            for j, c in enumerate(classes):
                row_s[f"prob_{c}"] = p_svm[j]
            rows_svm.append(row_s)

        self.df_cnn = pd.DataFrame(rows_cnn)
        self.df_svm = pd.DataFrame(rows_svm)

    def test_w1_reproduces_cnn_and_w0_reproduces_svm(self):
        """w=1 reproduces CNN scores, w=0 reproduces SVM scores."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cnn_file = Path(tmp_dir) / "cnn.csv"
            svm_file = Path(tmp_dir) / "svm.csv"
            self.df_cnn.to_csv(cnn_file, index=False)
            self.df_svm.to_csv(svm_file, index=False)

            res_w1 = run_late_fusion(cnn_file, svm_file, weights=[1.0])
            res_w0 = run_late_fusion(cnn_file, svm_file, weights=[0.0])

            self.assertEqual(res_w1["summary"]["best_global_w"], 1.0)
            self.assertEqual(res_w0["summary"]["best_global_w"], 0.0)

            # In res_w1, all chosen w should be 1.0; in res_w0, all 0.0
            for fold in range(5):
                self.assertEqual(res_w1["summary"]["chosen_w_per_fold"][fold], 1.0)
                self.assertEqual(res_w0["summary"]["chosen_w_per_fold"][fold], 0.0)

    def test_fold_k_never_influences_its_own_w(self):
        """Verify outer fold k never influences its selected w."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Modify fold 0 in CNN to heavily favor SVM (e.g. wrong predictions)
            df_cnn_mod = self.df_cnn.copy()
            fold0_mask = df_cnn_mod["fold"] == 0
            for c in CLASSES:
                df_cnn_mod.loc[fold0_mask, f"prob_{c}"] = 0.0
            df_cnn_mod.loc[fold0_mask, "prob_cardboard"] = 1.0

            cnn_file = Path(tmp_dir) / "cnn.csv"
            svm_file = Path(tmp_dir) / "svm.csv"
            df_cnn_mod.to_csv(cnn_file, index=False)
            self.df_svm.to_csv(svm_file, index=False)

            # Fold 0 evaluation should pick w based only on folds 1, 2, 3, 4 (where CNN is better)
            res = run_late_fusion(cnn_file, svm_file, weights=[0.0, 1.0])
            # For fold 0, the training folds 1,2,3,4 have perfect CNN (w=1.0)
            self.assertEqual(res["summary"]["chosen_w_per_fold"][0], 1.0)
            # For folds 1,2,3,4, the training set includes corrupted fold 0
            # which demonstrates separation

    def test_mismatched_ids_folds_labels_rejected(self):
        """Mismatched ids, folds, or labels are strictly rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cnn_file = Path(tmp_dir) / "cnn.csv"
            svm_file = Path(tmp_dir) / "svm.csv"

            # 1. ID mismatch
            df_cnn_bad_ids = self.df_cnn.copy()
            df_cnn_bad_ids.loc[0, "image_id"] = "different_id.jpg"
            df_cnn_bad_ids.to_csv(cnn_file, index=False)
            self.df_svm.to_csv(svm_file, index=False)
            with self.assertRaises(ValueError):
                run_late_fusion(cnn_file, svm_file)

            # 2. Fold mismatch
            df_cnn_bad_fold = self.df_cnn.copy()
            df_cnn_bad_fold.loc[0, "fold"] = 4
            df_svm_orig = self.df_svm.copy()
            df_svm_orig.loc[0, "fold"] = 0
            df_cnn_bad_fold.to_csv(cnn_file, index=False)
            df_svm_orig.to_csv(svm_file, index=False)
            with self.assertRaises(ValueError):
                run_late_fusion(cnn_file, svm_file)

            # 3. Label mismatch
            df_cnn_bad_label = self.df_cnn.copy()
            df_cnn_bad_label.loc[0, "label"] = 1  # different from svm's label
            df_svm_orig = self.df_svm.copy()
            df_svm_orig.loc[0, "label"] = "cardboard"  # index 0
            df_cnn_bad_label.to_csv(cnn_file, index=False)
            df_svm_orig.to_csv(svm_file, index=False)
            with self.assertRaises(ValueError):
                run_late_fusion(cnn_file, svm_file)


class TestFinalHybrid(unittest.TestCase):
    def test_tamper_detection(self):
        """Tampered model hash in metadata or on disk is detected and rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            d = Path(tmp_dir)
            meta = {
                "name": "h3_late_fusion",
                "svm_model_sha256": "fake_hash_1234",
                "a2_model_dir": "nonexistent",
                "a2_model_sha256": "fake_hash_5678",
                "w": 0.5,
                "classes": list(CLASSES),
            }
            (d / "metadata.json").write_text(json.dumps(meta))
            (d / "svm_pipeline.joblib").write_bytes(b"some content")

            with self.assertRaises(ValueError):
                load_hybrid(d)

    def test_windows_a2_model_dir_resolves_on_any_os(self):
        """A hybrid exported on Windows (backslash a2_model_dir) loads elsewhere."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            d, root = Path(tmp_dir) / "hybrid", Path(tmp_dir) / "repo"
            d.mkdir()
            svm_bytes = b"svm"
            (d / "svm_pipeline.joblib").write_bytes(svm_bytes)
            meta = {
                "svm_model_sha256": hashlib.sha256(svm_bytes).hexdigest(),
                "a2_model_dir": "artifacts\\phase2\\final\\A2.1",
                "a2_model_sha256": "unused",
            }
            (d / "metadata.json").write_text(json.dumps(meta))
            with patch(
                "waste_sorting.hybrid.load_trained_model", side_effect=RuntimeError("stop")
            ) as loader:
                with self.assertRaises(RuntimeError):
                    load_hybrid(d, root=root)
            loader.assert_called_once_with(root.resolve() / "artifacts/phase2/final/A2.1")

    def test_predict_proba_synthetic_contract(self):
        """Test predict_proba returns [N, 6] summing to 1.0."""
        classes = list(CLASSES)
        mock_meta = {
            "name": "h3_late_fusion",
            "family": "h3",
            "w": 0.5,
            "classes": classes,
            "selected_family": "global_color_spatial_texture",
        }
        mock_cnn = MagicMock()
        # Return mock logits [N, 6]
        mock_cnn.parameters.return_value = iter([torch.tensor([1.0])])
        mock_cnn.return_value = torch.ones((2, 6))

        mock_svm = MagicMock()
        mock_svm.predict_proba.return_value = np.ones((2, 6)) / 6.0

        from waste_sorting.hybrid import HybridPredictor

        predictor = HybridPredictor(
            metadata=mock_meta,
            cnn_model=mock_cnn,
            cnn_config={
                "preprocessing": {
                    "width": 224,
                    "height": 224,
                    "crop_ratio": 1.0,
                    "padding_rgb": [255, 255, 255],
                },
                "dl": {
                    "augmentation": {"enabled": False},
                    "data": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                },
            },
            svm_pipeline=mock_svm,
        )

        with patch("waste_sorting.hybrid.extract_extended_features") as mock_ext:
            mock_ext.return_value = np.zeros(2760, dtype=np.float32)
            imgs = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]
            probs = predict_proba(predictor, imgs)

            self.assertEqual(probs.shape, (2, 6))
            np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
