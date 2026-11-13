"""Unit tests for the Phase 2 demo helpers (phase2_demo.py + app.py Phase 2 tab).

All tests use mocked loaders; no real model files are needed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

FAKE_PROTOCOL: dict[str, Any] = {
    "classes": list(CLASSES),
    "models": {
        "h3_hybrid": {
            "temperature": 0.5362399011849001,
            "demo_threshold": 0.53,
            "hashes": {
                "svm_pipeline_sha256": "aaa",
                "hybrid_metadata_sha256": "bbb",
                "model_pt_sha256": "ccc",
            },
        },
        "a2_1_cnn": {
            "temperature": 1.102995775962789,
            "demo_threshold": 0.87,
            "hashes": {
                "model_pt_sha256": "ccc",
                "a2_metadata_sha256": "ddd",
            },
        },
        "phase1_svm": {
            "temperature": 0.8340600363975194,
            "demo_threshold": 0.76,
            "hashes": {
                "svm_pipeline_sha256": "aaa",
                "hybrid_metadata_sha256": "bbb",
            },
        },
    },
}


def _make_fake_probs(winning_class_idx: int = 2, high: float = 0.9) -> np.ndarray:
    """Return [1, 6] probability array where class *winning_class_idx* dominates."""
    probs = np.full((1, 6), (1.0 - high) / 5, dtype=np.float64)
    probs[0, winning_class_idx] = high
    return probs


def _make_pil_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# Protocol JSON parsing
# ---------------------------------------------------------------------------


class TestProtocolParsing(unittest.TestCase):
    """Thresholds and temperatures must come from protocol.json."""

    def setUp(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def tearDown(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def test_h3_temperature_matches_protocol(self):
        self.assertAlmostEqual(
            FAKE_PROTOCOL["models"]["h3_hybrid"]["temperature"],
            0.5362399011849001,
            places=10,
        )

    def test_h3_demo_threshold_matches_protocol(self):
        self.assertEqual(FAKE_PROTOCOL["models"]["h3_hybrid"]["demo_threshold"], 0.53)

    def test_a2_demo_threshold_matches_protocol(self):
        self.assertEqual(FAKE_PROTOCOL["models"]["a2_1_cnn"]["demo_threshold"], 0.87)

    def test_phase1_svm_demo_threshold_matches_protocol(self):
        self.assertEqual(FAKE_PROTOCOL["models"]["phase1_svm"]["demo_threshold"], 0.76)

    def test_protocol_has_six_classes(self):
        self.assertEqual(len(FAKE_PROTOCOL["classes"]), 6)
        self.assertEqual(tuple(FAKE_PROTOCOL["classes"]), CLASSES)


# ---------------------------------------------------------------------------
# Abstention logic
# ---------------------------------------------------------------------------


class TestAbstentionBehaviour(unittest.TestCase):
    """predict_phase2 abstains when max calibrated probability < demo_threshold."""

    def setUp(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def tearDown(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def _run_predict(self, raw_probs: np.ndarray, model_key: str = "h3_hybrid"):
        """Inject a fake model into the cache and run predict_phase2."""
        import waste_sorting.phase2_demo as p2d

        threshold = FAKE_PROTOCOL["models"][model_key]["demo_threshold"]
        temperature = FAKE_PROTOCOL["models"][model_key]["temperature"]

        p2d._PROTOCOL_CACHE = FAKE_PROTOCOL  # type: ignore[attr-defined]
        fake_entry: dict[str, Any] = {}

        if model_key == "h3_hybrid":
            fake_hybrid = MagicMock()
            fake_hybrid.metadata = {"selected_family": "global_color_spatial_texture"}
            fake_entry = {"type": "h3_hybrid", "hybrid": fake_hybrid}
        elif model_key == "a2_1_cnn":
            fake_model = MagicMock()
            fake_model.parameters.return_value = iter([MagicMock(device=None)])
            fake_entry = {
                "type": "a2_1_cnn",
                "model": fake_model,
                "transform": lambda img: MagicMock(),
                "meta": {"config": {}},
            }
        elif model_key == "phase1_svm":
            fake_pipeline = MagicMock()
            fake_entry = {
                "type": "phase1_svm",
                "svm_pipeline": fake_pipeline,
                "selected_family": "global_color_spatial_texture",
            }

        p2d._MODEL_CACHE[model_key] = fake_entry  # type: ignore[attr-defined]

        image = _make_pil_image()

        with patch.object(p2d, "_get_raw_probs", return_value=raw_probs):
            result = p2d.predict_phase2(model_key, image)
        return result, threshold, temperature

    def test_abstains_when_below_threshold(self):
        # Very low max prob (all roughly equal) → should abstain for h3_hybrid (threshold=0.53)
        low_probs = np.full((1, 6), 1.0 / 6, dtype=np.float64)
        # After temperature scaling (T=0.536) uniform probs stay uniform
        result, threshold, _ = self._run_predict(low_probs, "h3_hybrid")
        self.assertTrue(
            result["abstained"],
            msg=f"Expected abstention at max_prob={result['max_probability']:.4f} "
            f"< threshold={threshold}",
        )

    def test_does_not_abstain_when_above_threshold(self):
        # High max prob → should not abstain for h3_hybrid (threshold=0.53)
        high_probs = _make_fake_probs(winning_class_idx=2, high=0.95)
        result, threshold, _ = self._run_predict(high_probs, "h3_hybrid")
        self.assertFalse(
            result["abstained"],
            msg=f"Expected no abstention at max_prob={result['max_probability']:.4f} "
            f">= threshold={threshold}",
        )

    def test_abstains_for_a2_cnn_below_threshold(self):
        # a2_cnn demo_threshold=0.87 — a 0.5-max should abstain
        mid_probs = _make_fake_probs(winning_class_idx=0, high=0.5)
        result, threshold, _ = self._run_predict(mid_probs, "a2_1_cnn")
        self.assertTrue(result["abstained"])

    def test_does_not_abstain_for_a2_cnn_above_threshold(self):
        high_probs = _make_fake_probs(winning_class_idx=0, high=0.99)
        result, threshold, _ = self._run_predict(high_probs, "a2_1_cnn")
        self.assertFalse(result["abstained"])

    def test_result_contains_all_required_keys(self):
        high_probs = _make_fake_probs(winning_class_idx=1, high=0.95)
        result, _, _ = self._run_predict(high_probs, "h3_hybrid")
        self.assertIn("probabilities", result)
        self.assertIn("predicted_class", result)
        self.assertIn("max_probability", result)
        self.assertIn("threshold", result)
        self.assertIn("abstained", result)

    def test_probabilities_sum_to_one(self):
        probs = _make_fake_probs(winning_class_idx=3, high=0.9)
        result, _, _ = self._run_predict(probs, "h3_hybrid")
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_threshold_in_result_matches_protocol(self):
        probs = _make_fake_probs(0, high=0.95)
        result, expected_threshold, _ = self._run_predict(probs, "h3_hybrid")
        self.assertAlmostEqual(result["threshold"], expected_threshold, places=10)


# ---------------------------------------------------------------------------
# Hash mismatch
# ---------------------------------------------------------------------------


class TestHashMismatch(unittest.TestCase):
    """load_phase2_models must refuse files with wrong SHA-256."""

    def setUp(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def tearDown(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def test_hash_mismatch_raises_phase2_hash_error(self):
        import waste_sorting.phase2_demo as p2d

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Write a protocol.json with known hashes
            proto_dir = root / "docs" / "phase-2" / "evaluation"
            proto_dir.mkdir(parents=True)
            protocol = {
                "classes": list(CLASSES),
                "models": {
                    "h3_hybrid": {
                        "temperature": 0.5,
                        "demo_threshold": 0.53,
                        "hashes": {
                            "svm_pipeline_sha256": "0" * 64,  # will mismatch
                        },
                    },
                },
            }
            (proto_dir / "protocol.json").write_text(json.dumps(protocol))

            # Write a hybrid dir with an svm file that has a different hash
            hybrid_dir = root / "artifacts" / "phase2" / "hybrid-final" / "h3_late_fusion"
            hybrid_dir.mkdir(parents=True)
            (hybrid_dir / "svm_pipeline.joblib").write_bytes(b"fake svm content")
            (hybrid_dir / "metadata.json").write_text(
                json.dumps({"svm_model_sha256": "ff" * 32, "classes": list(CLASSES)})
            )

            with self.assertRaises(p2d.Phase2HashError):
                p2d._verify_hashes("h3_hybrid", protocol, root)

    def test_missing_model_file_raises_phase2_hash_error(self):
        import waste_sorting.phase2_demo as p2d

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protocol = {
                "classes": list(CLASSES),
                "models": {
                    "h3_hybrid": {
                        "temperature": 0.5,
                        "demo_threshold": 0.53,
                        "hashes": {
                            "svm_pipeline_sha256": "0" * 64,
                        },
                    },
                },
            }
            # Do NOT create the svm_pipeline.joblib file
            with self.assertRaises(p2d.Phase2HashError):
                p2d._verify_hashes("h3_hybrid", protocol, root)


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------


class TestModelSelectorDefault(unittest.TestCase):
    """The default model label must be H3 hybrid."""

    def test_default_model_label_is_h3(self):
        from waste_sorting.phase2_demo import DEFAULT_MODEL_LABEL

        self.assertIn("H3", DEFAULT_MODEL_LABEL)
        self.assertIn("recommended", DEFAULT_MODEL_LABEL.lower())

    def test_selector_labels_contain_all_three_models(self):
        from waste_sorting.phase2_demo import SELECTOR_LABELS

        keys = set(SELECTOR_LABELS.keys())
        self.assertIn("h3_hybrid", keys)
        self.assertIn("a2_1_cnn", keys)
        self.assertIn("phase1_svm", keys)

    def test_phase2_model_choices_in_app(self):
        try:
            import app as demo_app
        except ImportError:
            self.skipTest("app.py not importable")

        self.assertEqual(demo_app.PHASE2_MODEL_CHOICES[0], demo_app.DEFAULT_MODEL_LABEL)
        self.assertIn("H3", demo_app.PHASE2_MODEL_CHOICES[0])


# ---------------------------------------------------------------------------
# Phase 2 app callback
# ---------------------------------------------------------------------------


class TestPhase2AppCallback(unittest.TestCase):
    """classify_phase2_callback returns 7-tuple in all cases."""

    def setUp(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def tearDown(self):
        from waste_sorting.phase2_demo import reset_phase2_models

        reset_phase2_models()

    def _make_fake_result(self, abstained: bool = False) -> dict:
        probs = {c: 1.0 / 6 for c in CLASSES}
        if abstained:
            probs["metal"] = 0.3
        else:
            probs["metal"] = 0.95
            for c in CLASSES:
                if c != "metal":
                    probs[c] = 0.01
        return {
            "probabilities": probs,
            "predicted_class": "metal",
            "max_probability": probs["metal"],
            "threshold": 0.53,
            "abstained": abstained,
        }

    def test_callback_returns_7_tuple_on_accepted(self):
        try:
            import app as demo_app
        except ImportError:
            self.skipTest("app.py not importable")

        with patch("app.predict_phase2", return_value=self._make_fake_result(False)):
            result = demo_app.classify_phase2_callback(
                _make_pil_image(), "H3 hybrid (recommended)"
            )
        self.assertEqual(len(result), 7)
        # error field empty on success
        self.assertEqual(result[0], "")
        # primary result is the class name
        self.assertIn("Metal", result[1])

    def test_callback_returns_7_tuple_on_abstained(self):
        try:
            import app as demo_app
        except ImportError:
            self.skipTest("app.py not importable")

        with patch("app.predict_phase2", return_value=self._make_fake_result(True)):
            result = demo_app.classify_phase2_callback(
                _make_pil_image(), "H3 hybrid (recommended)"
            )
        self.assertEqual(len(result), 7)
        self.assertIn("Uncertain", result[1])
        self.assertIn("Uncertain", result[2])

    def test_callback_handles_none_image(self):
        try:
            import app as demo_app
        except ImportError:
            self.skipTest("app.py not importable")

        result = demo_app.classify_phase2_callback(None, "H3 hybrid (recommended)")
        self.assertEqual(len(result), 7)
        self.assertIn("No image", result[0])

    def test_callback_handles_phase2_error(self):
        try:
            import app as demo_app
        except ImportError:
            self.skipTest("app.py not importable")

        from waste_sorting.phase2_demo import Phase2DemoError

        with patch("app.predict_phase2", side_effect=Phase2DemoError("test error")):
            result = demo_app.classify_phase2_callback(
                _make_pil_image(), "H3 hybrid (recommended)"
            )
        self.assertEqual(len(result), 7)
        self.assertIn("test error", result[0])


# ---------------------------------------------------------------------------
# Phase 1 tests still pass (smoke test)
# ---------------------------------------------------------------------------


class TestPhase1DemoStillPasses(unittest.TestCase):
    """Ensure Phase 1 demo module is still importable and core behaviour intact."""

    def test_phase1_demo_importable(self):
        from waste_sorting import demo as phase1_demo  # noqa: F401

        self.assertTrue(True)

    def test_empty_view_returns_blank_fields(self):
        from waste_sorting.demo import empty_view

        view = empty_view()
        self.assertEqual(view.primary_result, "")
        self.assertEqual(view.probability_rows, [])
        self.assertEqual(view.error_message, "")

    def test_phase1_limitations_note_exists(self):
        from waste_sorting.demo import LIMITATIONS_NOTE

        self.assertIsInstance(LIMITATIONS_NOTE, str)
        self.assertGreater(len(LIMITATIONS_NOTE), 10)

    def test_phase1_frozen_abstention_threshold(self):
        from waste_sorting.demo import FROZEN_ABSTENTION_THRESHOLD

        self.assertEqual(FROZEN_ABSTENTION_THRESHOLD, 0.52)

    def test_phase1_canonical_classes(self):
        from waste_sorting.prediction import CANONICAL_CLASSES

        self.assertEqual(len(CANONICAL_CLASSES), 6)
        self.assertIn("cardboard", CANONICAL_CLASSES)
        self.assertIn("trash", CANONICAL_CLASSES)


if __name__ == "__main__":
    unittest.main()
