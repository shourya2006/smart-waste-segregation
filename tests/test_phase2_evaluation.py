"""Synthetic unit tests for Phase 2 calibration, validation thresholds, and protocol freeze."""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from waste_sorting.acquisition import sha256
from waste_sorting.dl_training import build_model
from waste_sorting.phase2_evaluation import (
    apply_temperature_scaling,
    calibrate_and_freeze,
    compute_bootstrap_ci,
    compute_gradcam_for_image,
    compute_pairwise_mcnemar,
    fit_temperature,
    load_and_validate_oof,
    main,
    predict_split,
    run_test_evaluation,
)

REPO = Path(__file__).resolve().parents[1]


class TemperatureScalingTests(unittest.TestCase):
    def test_temperature_scaling_lowers_log_loss_and_preserves_argmax(self):
        rng = np.random.default_rng(7)
        n_samples, n_classes = 120, 6
        y_true = rng.integers(0, n_classes, size=n_samples)

        # Deliberately over-confident logits -> spiky probabilities with ~80% accuracy
        logits = rng.normal(0.0, 1.0, size=(n_samples, n_classes))
        correct_mask = rng.random(n_samples) < 0.80
        for i in range(n_samples):
            if correct_mask[i]:
                logits[i, y_true[i]] += 4.5
            else:
                wrong = (y_true[i] + 1) % n_classes
                logits[i, wrong] += 4.5

        logits *= 5.0  # artificial over-confidence
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        nll_before = log_loss(y_true, probs, labels=list(range(n_classes)))
        temp = fit_temperature(y_true, probs)
        scaled = apply_temperature_scaling(probs, temp)
        nll_after = log_loss(y_true, scaled, labels=list(range(n_classes)))

        self.assertGreater(temp, 1.0)
        self.assertLess(nll_after, nll_before)
        np.testing.assert_array_equal(np.argmax(probs, axis=1), np.argmax(scaled, axis=1))
        self.assertTrue(np.allclose(scaled.sum(axis=1), 1.0))

    def test_temperature_scaling_clips_zeros_safely(self):
        y_true = np.array([0, 1, 2])
        probs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        temp = fit_temperature(y_true, probs)
        scaled = apply_temperature_scaling(probs, temp)
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue(np.allclose(scaled.sum(axis=1), 1.0))


class Phase2EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        # Provenance expects requirements.lock and src/waste_sorting under root
        (self.root / "requirements.lock").write_text("synthetic test fixture\n", encoding="utf-8")
        (self.root / "src/waste_sorting").mkdir(parents=True)
        (self.root / "src/waste_sorting/__init__.py").write_text("", encoding="utf-8")

        config = tomllib.loads((REPO / "configs/phase2.toml").read_text(encoding="utf-8"))
        config["preprocessing"].update(width=64, height=64)
        config["dl"]["data"].update(batch_size=4, num_workers=0)
        config["dl"]["training"].update(unfreeze_blocks=2)
        self.config = config
        self.classes = tuple(config["classes"])

        (self.root / "configs").mkdir()
        # Write minimal TOML matching config
        toml_lines = [
            f"classes = {json.dumps(list(self.classes))}",
            "[paths]",
            'image_root = "data/raw/trashnet"',
            'manifest = "docs/dataset/manifest.csv"',
            'folds = "docs/training/folds.json"',
            'artifacts = "artifacts/phase2"',
            "[preprocessing]",
            "width = 64",
            "height = 64",
            "preserve_aspect_ratio = true",
            "padding_rgb = [255, 255, 255]",
            "[dl.data]",
            "batch_size = 4",
            "num_workers = 0",
            "seed = 42",
            "mean = [0.485, 0.456, 0.406]",
            "std = [0.229, 0.224, 0.225]",
            "balanced_sampling = false",
            "[dl.augmentation]",
            "enabled = false",
            "[dl.training]",
            'backbone = "mobilenet_v2"',
            "unfreeze_blocks = 2",
            "dropout = 0.2",
        ]
        (self.root / "configs/phase2.toml").write_text(
            "\n".join(toml_lines) + "\n", encoding="utf-8"
        )

        # Synthetic dataset across train (12), validation (6), test (6)
        rows = []
        rng = np.random.default_rng(123)
        for label in self.classes:
            for split, count in (("train", 2), ("validation", 1), ("test", 1)):
                for idx in range(count):
                    image_id = f"{label}/{split}_{idx}.png"
                    rows.append(
                        {
                            "image_id": image_id,
                            "label": label,
                            "included": True,
                            "duplicate_group": f"grp_{label}_{split}_{idx}",
                            "split": split,
                        }
                    )
                    img_path = self.root / "data/raw/trashnet" / image_id
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    arr = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(img_path)

        manifest_df = pd.DataFrame(rows)
        (self.root / "docs/dataset").mkdir(parents=True)
        manifest_df.to_csv(self.root / "docs/dataset/manifest.csv", index=False)

        train_ids = sorted(manifest_df.loc[manifest_df["split"] == "train", "image_id"])
        folds = {img_id: i % 2 for i, img_id in enumerate(train_ids)}
        (self.root / "docs/training").mkdir(parents=True)
        (self.root / "docs/training/folds.json").write_text(
            json.dumps({"n_splits": 2, "assignments": folds}),
            encoding="utf-8",
        )

        # Synthetic A2.1 CNN (random weights, pretrained=False, no network)
        import torch

        torch.manual_seed(0)
        a2_dir = self.root / "artifacts/phase2/final/A2.1"
        a2_dir.mkdir(parents=True)
        cnn = build_model(self.config, pretrained=False)
        model_pt = a2_dir / "model.pt"
        torch.save(cnn.state_dict(), model_pt)
        a2_sha = sha256(model_pt)
        a2_meta = {
            "mode": "final",
            "epochs": 1,
            "classes": list(self.classes),
            "config": self.config,
            "model_sha256": a2_sha,
        }
        (a2_dir / "metadata.json").write_text(json.dumps(a2_meta), encoding="utf-8")

        # Synthetic SVM pipeline (DummyClassifier returning deterministic probabilities)
        hybrid_dir = self.root / "artifacts/phase2/hybrid-final/h3_late_fusion"
        hybrid_dir.mkdir(parents=True)
        dummy = DummyClassifier(strategy="uniform", random_state=42)
        dummy.fit(
            np.zeros((len(self.classes), 1608), dtype=np.float32),
            np.arange(len(self.classes)),
        )
        svm_pipe = Pipeline([("classifier", dummy)])
        svm_path = hybrid_dir / "svm_pipeline.joblib"
        joblib.dump(svm_pipe, svm_path)
        svm_sha = sha256(svm_path)

        hybrid_meta = {
            "name": "h3_late_fusion",
            "family": "h3",
            "w": 0.5,
            "classes": list(self.classes),
            "svm_model_sha256": svm_sha,
            "a2_model_dir": "artifacts\\phase2\\final\\A2.1",
            "a2_model_sha256": a2_sha,
            "selected_family": "global_color_spatial_texture",
        }
        (hybrid_dir / "metadata.json").write_text(json.dumps(hybrid_meta), encoding="utf-8")

        # Synthetic training OOF files
        self.train_df = manifest_df.loc[manifest_df["split"] == "train"].sort_values("image_id")
        self.val_df = manifest_df.loc[manifest_df["split"] == "validation"].sort_values("image_id")
        self.cnn_oof = self.root / "docs/phase-2/ablations/selected-oof_predictions.csv"
        self.svm_oof = self.root / "artifacts/phase2/eval/phase1_svm_oof.csv"
        self.h3_oof = self.root / "artifacts/phase2/eval/h3/oof_predictions.csv"
        self._write_oof(self.cnn_oof, integer_labels=True)
        self._write_oof(self.svm_oof, integer_labels=False)
        self._write_oof(self.h3_oof, integer_labels=False)

    def _write_oof(self, path: Path, *, integer_labels: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, row in enumerate(self.train_df.itertuples()):
            true_idx = self.classes.index(row.label)
            probs = np.full(len(self.classes), 0.02, dtype=np.float64)
            probs[true_idx] = 0.90
            rec = {
                "image_id": row.image_id,
                "fold": idx % 2,
                "label": true_idx if integer_labels else row.label,
            }
            for c_idx, c_name in enumerate(self.classes):
                rec[f"prob_{c_name}"] = float(probs[c_idx])
            rows.append(rec)
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_predict_split_refuses_test_without_protocol_and_allows_validation(self):
        for model_name in ("phase1_svm", "a2_1_cnn", "h3_hybrid"):
            with self.assertRaises(FileNotFoundError):
                predict_split(model_name, "test", self.root)

        ids_svm, y_svm, p_svm = predict_split("phase1_svm", "validation", self.root)
        ids_cnn, y_cnn, p_cnn = predict_split("a2_1_cnn", "validation", self.root)
        ids_h3, y_h3, p_h3 = predict_split("h3_hybrid", "validation", self.root)

        self.assertEqual(len(ids_svm), 6)
        np.testing.assert_array_equal(ids_svm, ids_cnn)
        np.testing.assert_array_equal(ids_cnn, ids_h3)
        np.testing.assert_array_equal(y_svm, y_cnn)
        np.testing.assert_array_equal(y_cnn, y_h3)
        self.assertEqual(p_h3.shape, (6, 6))
        np.testing.assert_allclose(p_h3, 0.5 * p_cnn + 0.5 * p_svm, atol=1e-6)

    def test_oof_validation_rejects_wrong_extra_or_missing_ids(self):
        df = pd.read_csv(self.svm_oof)

        # Extra row
        extra_path = self.root / "artifacts/extra.csv"
        extra_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        extra_df.to_csv(extra_path, index=False)
        with self.assertRaises(ValueError):
            load_and_validate_oof(extra_path, self.root, self.classes)

        # Wrong ID (e.g. validation image_id instead of training image_id)
        wrong_path = self.root / "artifacts/wrong.csv"
        wrong_df = df.copy()
        wrong_df.loc[0, "image_id"] = self.val_df["image_id"].iloc[0]
        wrong_df.to_csv(wrong_path, index=False)
        with self.assertRaises(ValueError):
            load_and_validate_oof(wrong_path, self.root, self.classes)

        # Missing row
        missing_path = self.root / "artifacts/missing.csv"
        df.iloc[:-1].to_csv(missing_path, index=False)
        with self.assertRaises(ValueError):
            load_and_validate_oof(missing_path, self.root, self.classes)

        # Extra distinct ID
        extra_distinct = self.root / "artifacts/extra_distinct.csv"
        extra_row = df.iloc[[0]].copy()
        extra_row.loc[:, "image_id"] = self.val_df["image_id"].iloc[0]
        pd.concat([df, extra_row], ignore_index=True).to_csv(extra_distinct, index=False)
        with self.assertRaises(ValueError):
            load_and_validate_oof(extra_distinct, self.root, self.classes)

    def test_calibrate_creates_protocol_refuses_overwrite_and_unlocks_test_split(self):
        out_dir = self.root / "docs/phase-2/evaluation"
        main(["calibrate", "--root", str(self.root), "--output", str(out_dir)])

        proto_path = out_dir / "protocol.json"
        self.assertTrue(proto_path.is_file())
        protocol = json.loads(proto_path.read_text(encoding="utf-8"))
        self.assertEqual(list(protocol["classes"]), list(self.classes))
        self.assertIn(
            "demo_threshold added before the freeze",
            protocol["abstention"]["note"],
        )
        self.assertIn("limitations", protocol)
        self.assertIn(
            "demo_threshold",
            protocol["planned_test_analyses"]["selective_metrics"]["thresholds_reported"],
        )
        for model_name in ("phase1_svm", "a2_1_cnn", "h3_hybrid"):
            self.assertIn(model_name, protocol["models"])
            model_entry = protocol["models"][model_name]
            self.assertIn("headline_threshold", model_entry)
            self.assertIn("demo_threshold", model_entry)
            self.assertIn("headline_threshold_selection", model_entry)
            self.assertIn("demo_threshold_selection", model_entry)
            self.assertTrue((out_dir / f"validation-threshold-sweep-{model_name}.csv").is_file())
            self.assertTrue((out_dir / f"validation-reliability-{model_name}.png").is_file())

        # Overwrite must be refused
        with self.assertRaises(FileExistsError):
            calibrate_and_freeze(root=self.root, output_dir=out_dir)

        # Test split is now unlocked and verifies hashes
        ids_test, y_test, p_test = predict_split("h3_hybrid", "test", self.root)
        self.assertEqual(len(ids_test), 6)
        self.assertEqual(p_test.shape, (6, 6))

    def test_bootstrap_ci_deterministic_and_bounded(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 5])

        ci1 = compute_bootstrap_ci(y_true, y_pred, n_resamples=200, seed=42)
        ci2 = compute_bootstrap_ci(y_true, y_pred, n_resamples=200, seed=42)
        self.assertEqual(ci1, ci2)
        for metric_key in ("macro_f1", "accuracy"):
            m = ci1[metric_key]
            self.assertGreaterEqual(m["ci_low"], 0.0)
            self.assertLessEqual(m["ci_low"], m["point_estimate"])
            self.assertLessEqual(m["point_estimate"], m["ci_high"])
            self.assertLessEqual(m["ci_high"], 1.0)

        with self.assertRaises(ValueError):
            compute_bootstrap_ci(np.array([]), np.array([]))

    def test_pairwise_mcnemar_counts_and_pvalues(self):
        # 20 samples: both_ok=7, a_only=10, b_only=1, both_err=2 -> p = binomtest(1, 11, 0.5)
        y_true = np.zeros(20, dtype=np.int64)
        pred_a = np.array([0] * 7 + [0] * 10 + [1] * 1 + [1] * 2, dtype=np.int64)
        pred_b = np.array([0] * 7 + [1] * 10 + [0] * 1 + [1] * 2, dtype=np.int64)

        res = compute_pairwise_mcnemar(y_true, pred_a, pred_b, bonferroni_comparisons=3)
        self.assertEqual(res["both_correct"], 7)
        self.assertEqual(res["a_only_correct"], 10)
        self.assertEqual(res["b_only_correct"], 1)
        self.assertEqual(res["both_wrong"], 2)
        self.assertEqual(res["discordant_total"], 11)
        self.assertAlmostEqual(res["bonferroni_alpha"], 0.05 / 3.0)
        # binomtest(1, 11, 0.5).pvalue = 12 / 1024 = 0.01171875 < 0.01667
        self.assertAlmostEqual(res["p_value"], 12.0 / 1024.0, places=10)
        self.assertTrue(res["significant_unadjusted"])
        self.assertTrue(res["significant_bonferroni"])

        identical = compute_pairwise_mcnemar(y_true, pred_a, pred_a)
        self.assertEqual(identical["discordant_total"], 0)
        self.assertEqual(identical["p_value"], 1.0)
        self.assertFalse(identical["significant_bonferroni"])

    def test_gradcam_heatmap_shape_range_and_finite(self):
        import torch

        torch.manual_seed(0)
        model = build_model(self.config, pretrained=False)
        img_tensor = torch.randn(3, 64, 64, dtype=torch.float32)
        cam = compute_gradcam_for_image(model, img_tensor, target_class_idx=1)

        self.assertEqual(cam.shape, (64, 64))
        self.assertTrue(np.isfinite(cam).all())
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)

    def test_test_evaluation_cli_guards_hashes_overwrite_and_writes_artifacts(self):
        out_dir = self.root / "docs/phase-2/evaluation"
        main(["calibrate", "--root", str(self.root), "--output", str(out_dir)])
        main(["test", "--root", str(self.root), "--output", str(out_dir)])

        pred_csv = out_dir / "test-predictions.csv"
        metrics_json = out_dir / "test-metrics.json"
        self.assertTrue(pred_csv.is_file())
        self.assertTrue(metrics_json.is_file())
        self.assertTrue((out_dir / "test-gradcam-a2_1_cnn.png").is_file())

        pred_df = pd.read_csv(pred_csv)
        self.assertEqual(len(pred_df), 6)
        for model_name in ("phase1_svm", "a2_1_cnn", "h3_hybrid"):
            self.assertIn(f"{model_name}_abstained_headline", pred_df.columns)
            self.assertIn(f"{model_name}_abstained_demo", pred_df.columns)
            self.assertTrue((out_dir / f"test-confusion-{model_name}.png").is_file())
            self.assertTrue((out_dir / f"test-reliability-{model_name}.png").is_file())

        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        self.assertEqual(metrics["sample_count"], 6)
        self.assertEqual(metrics["robustness"], "robustness set not available")
        self.assertEqual(
            set(metrics["mcnemar"].keys()),
            {
                "h3_hybrid_vs_a2_1_cnn",
                "h3_hybrid_vs_phase1_svm",
                "a2_1_cnn_vs_phase1_svm",
            },
        )
        for model_name in ("phase1_svm", "a2_1_cnn", "h3_hybrid"):
            m = metrics["models"][model_name]
            self.assertIn("selective_headline", m)
            self.assertIn("selective_demo", m)
            self.assertIn("bootstrap_95_ci", m)

        # Report generation from saved files only (no inference)
        latex_out = self.root / "report/evaluation-section.tex"
        main(
            [
                "report",
                "--root",
                str(self.root),
                "--output",
                str(out_dir),
                "--latex",
                str(latex_out),
            ]
        )
        readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
        tex_text = latex_out.read_text(encoding="utf-8")
        self.assertIn("Headline (90% val target)", readme_text)
        self.assertIn("Demo (95% val target)", readme_text)
        self.assertIn("robustness set not available", readme_text)
        self.assertIn("Pairwise Exact McNemar Tests", readme_text)
        self.assertIn("tab:phase2_main", tex_text)
        self.assertIn("tab:phase2_mcnemar", tex_text)

        # Overwrite must be refused
        with self.assertRaises(FileExistsError):
            run_test_evaluation(root=self.root, output_dir=out_dir, allow_overwrite=False)

        # Tampering with frozen artifact must fail hash verification
        pred_csv.unlink()
        model_pt = self.root / "artifacts/phase2/final/A2.1/model.pt"
        model_pt.write_bytes(b"corrupted-checkpoint")
        with self.assertRaises(ValueError):
            run_test_evaluation(root=self.root, output_dir=out_dir)


if __name__ == "__main__":
    unittest.main()
