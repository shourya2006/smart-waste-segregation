"""Integration tests for the photo demo, using synthetic fixtures."""

from __future__ import annotations

import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from waste_sorting.demo import (
    CLASS_DISPLAY_LABELS,
    DEFAULT_ARTIFACT_PATH,
    UNCERTAIN_PRIMARY,
    ArtifactIntegrityError,
    DemoError,
    DemoView,
    artifact_load_count,
    classify_upload,
    empty_view,
    evaluated_artifact_status,
    format_prediction,
    inject_runtime_artifact,
    load_evaluated_artifact,
    reset_runtime_artifact,
    verify_artifact_file,
)
from waste_sorting.extended_features import DIMENSION as EXTENDED_DIMENSION
from waste_sorting.prediction import CANONICAL_CLASSES, PredictionResult, predict_image


class DummyCalibratedModel:
    def __init__(self, probabilities=None):
        self.classes_ = np.arange(len(CANONICAL_CLASSES))
        self.n_features_in_ = EXTENDED_DIMENSION
        self.probabilities = (
            probabilities
            if probabilities is not None
            else np.array([0.04, 0.03, 0.78, 0.05, 0.06, 0.04], dtype=np.float64)
        )

    def predict_proba(self, features):
        return np.tile(self.probabilities, (len(features), 1))


def accepted_artifact(threshold=0.52):
    return {
        "calibrated_model": DummyCalibratedModel(),
        "classes": list(CANONICAL_CLASSES),
        "threshold": threshold,
        "input_dimension": EXTENDED_DIMENSION,
    }


def abstained_artifact():
    probs = np.array([0.18, 0.16, 0.17, 0.19, 0.15, 0.15], dtype=np.float64)
    return {
        "calibrated_model": DummyCalibratedModel(probs),
        "classes": list(CANONICAL_CLASSES),
        "threshold": 0.52,
        "input_dimension": EXTENDED_DIMENSION,
    }


class DemoPredictionTests(unittest.TestCase):
    def setUp(self):
        reset_runtime_artifact()
        self.image = Image.new("RGB", (48, 40), color=(200, 120, 60))

    def tearDown(self):
        reset_runtime_artifact()

    def test_callback_uses_shared_predict_image(self):
        inject_runtime_artifact(accepted_artifact())
        with patch("waste_sorting.demo.predict_image", wraps=predict_image) as mocked:
            view = classify_upload(self.image)
        mocked.assert_called_once()
        args, kwargs = mocked.call_args
        self.assertIsInstance(args[0], Image.Image)
        self.assertIn("calibrated_model", args[1])
        self.assertIsInstance(view, DemoView)
        self.assertTrue(view.used_shared_interface)

    def test_accepted_labels_confidence_and_six_class_display(self):
        inject_runtime_artifact(accepted_artifact())
        view = classify_upload(self.image)
        self.assertEqual(view.primary_result, "Metal")
        self.assertEqual(view.uncertainty_message, "")
        self.assertIn("Calibrated model confidence: 78.0%", view.confidence_text)
        self.assertIn("0.52", view.confidence_text)
        self.assertEqual(len(view.probability_rows), 6)
        self.assertEqual(
            [row[0] for row in view.probability_rows],
            [CLASS_DISPLAY_LABELS[name] for name in CANONICAL_CLASSES],
        )
        self.assertEqual(view.probability_rows[2], ["Metal", "78.00%"])
        self.assertEqual(view.tentative_class, "")
        self.assertEqual(view.error_message, "")

    def test_abstained_presentation_uses_uncertainty_as_primary(self):
        inject_runtime_artifact(abstained_artifact())
        view = classify_upload(self.image)
        self.assertEqual(view.primary_result, UNCERTAIN_PRIMARY)
        self.assertEqual(view.uncertainty_message, UNCERTAIN_PRIMARY)
        self.assertIn("Calibrated model confidence:", view.confidence_text)
        self.assertIn("Tentative top class (not accepted): Paper", view.tentative_class)
        self.assertEqual(len(view.probability_rows), 6)

    def test_empty_upload_is_rejected(self):
        inject_runtime_artifact(accepted_artifact())
        with self.assertRaises(DemoError) as ctx:
            classify_upload(None)
        self.assertIn("No image uploaded", str(ctx.exception))

    def test_empty_numpy_array_is_rejected(self):
        inject_runtime_artifact(accepted_artifact())
        with self.assertRaises(DemoError) as ctx:
            classify_upload(np.array([]))
        self.assertIn("No image uploaded", str(ctx.exception))

    def test_corrupt_image_file_is_rejected(self):
        inject_runtime_artifact(accepted_artifact())
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.jpg"
            corrupt.write_bytes(b"this is not an image")
            with self.assertRaises(DemoError) as ctx:
                classify_upload(corrupt)
            self.assertRegex(str(ctx.exception), r"could not be read as an image")

    def test_oversized_image_is_rejected(self):
        inject_runtime_artifact(accepted_artifact())
        huge = Image.new("RGB", (32, 32), color=(10, 20, 30))
        with self.assertRaises(DemoError) as ctx:
            classify_upload(huge, max_pixels=100, max_side=16)
        self.assertIn("unreasonably large", str(ctx.exception))

    def test_incompatible_artifact_is_rejected(self):
        bad = accepted_artifact()
        bad["classes"] = ["paper", "plastic"]
        with self.assertRaises(DemoError) as ctx:
            classify_upload(self.image, artifact=bad)
        self.assertIn("incompatible", str(ctx.exception).lower())

    def test_missing_artifact_file_is_reported(self):
        missing = Path("artifacts/calibrated_svm/definitely-missing.joblib")
        with self.assertRaises(ArtifactIntegrityError) as ctx:
            load_evaluated_artifact(missing, expected_sha256="abc", expected_size=1)
        self.assertIn("was not found", str(ctx.exception))

    def test_checksum_mismatch_refuses_to_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            payload = b"not-the-evaluated-model"
            path.write_bytes(payload)
            with self.assertRaises(ArtifactIntegrityError) as ctx:
                verify_artifact_file(
                    path,
                    expected_sha256="0" * 64,
                    expected_size=len(payload),
                )
            self.assertIn("SHA-256 does not match", str(ctx.exception))
            with self.assertRaises(ArtifactIntegrityError):
                load_evaluated_artifact(
                    path,
                    expected_sha256="0" * 64,
                    expected_size=len(payload),
                )

    def test_size_mismatch_is_reported_before_unpickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            path.write_bytes(b"abcd")
            with patch("waste_sorting.demo.joblib.load") as loader:
                with self.assertRaises(ArtifactIntegrityError) as ctx:
                    load_evaluated_artifact(
                        path,
                        expected_sha256="0" * 64,
                        expected_size=99,
                    )
                self.assertIn("file size", str(ctx.exception))
                loader.assert_not_called()

    def test_stale_results_are_cleared(self):
        cleared = empty_view()
        self.assertEqual(cleared.primary_result, "")
        self.assertEqual(cleared.probability_rows, [])
        self.assertEqual(cleared.confidence_text, "")
        self.assertEqual(cleared.uncertainty_message, "")
        self.assertEqual(cleared.tentative_class, "")
        accepted = format_prediction(
            PredictionResult(
                predicted_class="metal",
                probabilities={name: 0.0 for name in CANONICAL_CLASSES}
                | {"metal": 1.0},
                max_probability=1.0,
                abstained=False,
            )
        )
        self.assertNotEqual(accepted.primary_result, "")
        self.assertEqual(empty_view().primary_result, "")

    def test_model_is_reused_across_requests(self):
        inject_runtime_artifact(accepted_artifact())
        self.assertEqual(artifact_load_count(), 1)
        classify_upload(self.image)
        classify_upload(Image.new("RGB", (32, 32), color=(30, 80, 140)))
        self.assertEqual(artifact_load_count(), 1)

    def test_numpy_rgb_array_is_accepted(self):
        inject_runtime_artifact(accepted_artifact())
        array = np.zeros((24, 24, 3), dtype=np.uint8)
        array[:, :] = (180, 90, 40)
        view = classify_upload(array)
        self.assertEqual(view.primary_result, "Metal")

    def test_evaluated_artifact_status_is_explicit(self):
        status = evaluated_artifact_status(DEFAULT_ARTIFACT_PATH)
        self.assertIn(status, {"verified", "missing", "checksum-mismatch"})

    @unittest.skipUnless(
        evaluated_artifact_status() == "verified",
        "preserved evaluated model artifact is missing or checksum-mismatched",
    )
    def test_evaluated_artifact_on_synthetic_image(self):
        reset_runtime_artifact()
        load_evaluated_artifact()
        view = classify_upload(self.image)
        self.assertEqual(len(view.probability_rows), 6)
        self.assertIn("Calibrated model confidence:", view.confidence_text)
        self.assertIn(view.primary_result, {*CLASS_DISPLAY_LABELS.values(), UNCERTAIN_PRIMARY})


@unittest.skipUnless(importlib.util.find_spec("gradio"), "install the demo dependencies")
class DemoAppConstructionTests(unittest.TestCase):
    def test_importing_app_does_not_launch_server(self):
        with patch("gradio.Blocks.launch") as launch:
            import app as demo_app

            importlib.reload(demo_app)
            launch.assert_not_called()

    def test_build_interface_does_not_launch_server(self):
        import app as demo_app

        with patch("gradio.Blocks.launch") as launch:
            interface = demo_app.build_interface()
            launch.assert_not_called()
        self.assertIsNotNone(interface)

    def test_clear_event_includes_upload_and_all_result_fields(self):
        import app as demo_app

        config = demo_app.build_interface().get_config_file()
        clear_event = next(d for d in config["dependencies"] if d["api_name"] == "clear")
        components = {c["id"]: c for c in config["components"]}
        self.assertEqual(len(clear_event["outputs"]), len(demo_app.clear_all()))
        self.assertEqual(components[clear_event["outputs"][0]]["type"], "image")

    def test_error_output_is_visible_and_read_only(self):
        import app as demo_app

        config = demo_app.build_interface().get_config_file()
        status = next(
            c for c in config["components"] if c["props"].get("elem_id") == "demo-status"
        )
        self.assertTrue(status["props"]["visible"])
        self.assertFalse(status["props"]["interactive"])

    def test_real_gradio_distribution_is_used(self):
        import gradio

        self.assertEqual(gradio.__version__, "6.30.0")
        self.assertNotEqual(
            Path(gradio.__file__).resolve().parent,
            Path(__file__).resolve().parents[1] / "gradio",
        )

    def test_image_change_and_clear_drop_stale_outputs(self):
        import app as demo_app

        inject_runtime_artifact(accepted_artifact())
        classified = demo_app.classify_callback(Image.new("RGB", (32, 32), (12, 24, 36)))
        self.assertEqual(classified[1], "Metal")
        cleared = demo_app.on_image_change(Image.new("RGB", (16, 16), (1, 2, 3)))
        self.assertEqual(cleared[1], "")
        self.assertEqual(cleared[4], [])
        reset = demo_app.clear_all()
        self.assertIsNone(reset[0])
        self.assertEqual(reset[2], "")

    def test_callback_empty_and_corrupt_inputs(self):
        import app as demo_app

        inject_runtime_artifact(accepted_artifact())
        empty = demo_app.classify_callback(None)
        self.assertIn("No image uploaded", empty[0])
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "bad.png"
            corrupt.write_bytes(b"not-png")
            failed = demo_app.classify_callback(str(corrupt))
        self.assertIn("could not be read as an image", failed[0])
        self.assertEqual(failed[1], "")

    def test_share_flag_is_rejected(self):
        import app as demo_app

        with self.assertRaises(SystemExit):
            demo_app.main(["--share"])


class DemoArtifactRoundTripTests(unittest.TestCase):
    def test_saved_fixture_joblib_round_trip(self):
        reset_runtime_artifact()
        artifact = accepted_artifact()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.joblib"
            joblib.dump(artifact, path)
            payload = path.read_bytes()
            digest = __import__("hashlib").sha256(payload).hexdigest()
            loaded = load_evaluated_artifact(
                path,
                expected_sha256=digest,
                expected_size=len(payload),
            )
            image = Image.new("RGB", (32, 32), (90, 40, 20))
            view = classify_upload(image, artifact=loaded)
            self.assertEqual(view.primary_result, "Metal")
            self.assertEqual(artifact_load_count(), 1)
            load_evaluated_artifact(
                path,
                expected_sha256=digest,
                expected_size=len(payload),
            )
            self.assertEqual(artifact_load_count(), 1)
        reset_runtime_artifact()


class UnusedImportGuard(unittest.TestCase):
    def test_bytes_io_corrupt_buffer(self):
        inject_runtime_artifact(accepted_artifact())
        buffer = io.BytesIO(b"not-an-image")
        with self.assertRaises(DemoError):
            classify_upload(buffer)


if __name__ == "__main__":
    unittest.main()
