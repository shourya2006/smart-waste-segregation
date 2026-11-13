"""Synthetic tests for the external robustness set: audit, isolation and single evaluation.

No real photos and no real models: images are random noise, the TrashNet manifest and the
frozen protocol are small synthetic stand-ins, and the predictor is a fake.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting import robustness_evaluation as rob
from waste_sorting.acquisition import CLASSES, sha256
from waste_sorting.data_cli import main as data_main
from waste_sorting.dl_data import load_manifest

REPO = Path(__file__).resolve().parents[1]
PER_CLASS = 3


def _noise(rng: np.random.Generator, size: int = 64) -> Image.Image:
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


class RobustnessFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rng = np.random.default_rng(2026)
        self.photos = self.root / "data/raw/robustness"
        for label in CLASSES:
            folder = self.photos / label
            folder.mkdir(parents=True)
            for index in range(PER_CLASS):
                _noise(self.rng).save(folder / f"{label}{index}.png")
        self._write_trashnet_records([])
        self._write_protocol()

    # -- synthetic project records -------------------------------------------------------

    def _write_trashnet_records(self, extra_rows):
        rows = []
        for index, label in enumerate(CLASSES):
            image = _noise(self.rng)
            rows.append(
                {
                    "image_id": f"{label}/{label}{index}.jpg",
                    "label": label,
                    "sha256": f"{index:064x}",
                    "phash": str(imagehash.phash(image)),
                    "included": True,
                    "duplicate_group": f"{label}/{label}{index}.jpg",
                    "split": "train",
                }
            )
        rows.extend(extra_rows)
        dataset = self.root / "docs/dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dataset / "manifest.csv", index=False)
        (dataset / "audit_settings.json").write_text(json.dumps({"phash_distance": 6}))

    def _write_protocol(self):
        evaluation = self.root / "docs/phase-2/evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        protocol = {
            "classes": list(CLASSES),
            "models": {
                name: {"temperature": 1.0, "demo_threshold": 0.6, "headline_threshold": 0.0}
                for name in rob.MODEL_NAMES
            },
            "planned_test_analyses": {
                "bootstrap": {"n_resamples": 50, "seed": 42},
                "pairwise_mcnemar": {"pairs": [list(p) for p in rob.DEFAULT_MCNEMAR_PAIRS]},
            },
        }
        self.protocol_path = evaluation / "protocol.json"
        self.protocol_path.write_text(json.dumps(protocol, indent=2))
        reference = {
            "unconditional": {
                "macro_f1": 0.92,
                "accuracy": 0.93,
                "per_class": {c: {"recall": 0.9} for c in CLASSES},
            },
            "selective_demo": {"coverage": 0.95, "accepted_accuracy": 0.95},
            "bootstrap_95_ci": {
                "macro_f1": {"point_estimate": 0.92, "ci_low": 0.89, "ci_high": 0.95},
                "accuracy": {"point_estimate": 0.93, "ci_low": 0.91, "ci_high": 0.96},
            },
        }
        self.test_metrics_path = evaluation / "test-metrics.json"
        self.test_metrics_path.write_text(
            json.dumps({"sample_count": 377, "models": {m: reference for m in rob.MODEL_NAMES}})
        )

    def audit(self):
        with patch("sys.argv", ["waste-data", "--root", str(self.root), "robustness-audit"]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                data_main()
        return out.getvalue()

    def manifest(self):
        return pd.read_csv(
            self.root / "docs/robustness/manifest.csv", dtype=str, keep_default_na=False
        )

    def fill_conditions(self):
        path = self.root / "docs/robustness/conditions.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        backgrounds = rob.ALLOWED_CONDITIONS["background"]
        lightings = rob.ALLOWED_CONDITIONS["lighting"]
        frame["background"] = [backgrounds[i % len(backgrounds)] for i in range(len(frame))]
        frame["lighting"] = [lightings[i % len(lightings)] for i in range(len(frame))]
        frame["device"] = [f"phone_{'abc'[i % 3]}" for i in range(len(frame))]
        frame.to_csv(path, index=False)


def fake_predictor(calls):
    """Deterministic pseudo-predictions derived from pixel content (some right, some wrong)."""

    def predict(model_name, image, root="."):
        calls.append(model_name)
        seed = int(np.asarray(image).sum()) % 1000
        label = CLASSES[seed % len(CLASSES)]
        probs = {c: 0.02 for c in CLASSES}
        probs[label] = 0.9
        return {
            "probabilities": probs,
            "predicted_class": label,
            "max_probability": 0.9,
            "threshold": 0.6,
            "abstained": False,
        }

    return predict


class AuditTests(RobustnessFixture):
    def test_manifest_has_exact_columns_and_conditions_template_is_created(self):
        output = self.audit()
        manifest = self.manifest()
        self.assertEqual(tuple(manifest.columns), rob.MANIFEST_COLUMNS)
        self.assertEqual(len(manifest), len(CLASSES) * PER_CLASS)
        self.assertTrue((manifest["included"] == "True").all())
        self.assertTrue(manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertTrue(manifest["phash"].str.fullmatch(r"[0-9a-f]{16}").all())
        self.assertEqual(set(manifest["width"]), {"64"})
        conditions = pd.read_csv(self.root / "docs/robustness/conditions.csv", dtype=str)
        self.assertEqual(tuple(conditions.columns), rob.CONDITIONS_COLUMNS)
        self.assertEqual(sorted(conditions["image_id"]), sorted(manifest["image_id"]))
        self.assertIn("cardboard", output)
        self.assertIn("Conditions incomplete", output)

    def test_filled_conditions_are_copied_and_existing_rows_are_kept(self):
        self.audit()
        self.fill_conditions()
        before = pd.read_csv(self.root / "docs/robustness/conditions.csv", dtype=str)
        _noise(self.rng).save(self.photos / "paper/new.png")
        self.audit()
        after = pd.read_csv(
            self.root / "docs/robustness/conditions.csv", dtype=str, keep_default_na=False
        )
        pd.testing.assert_frame_equal(after.iloc[: len(before)], before)
        self.assertEqual(after.iloc[-1]["image_id"], "paper/new.png")
        manifest = self.manifest().set_index("image_id")
        self.assertEqual(
            manifest.loc["cardboard/cardboard0.png", "background"], before.iloc[0]["background"]
        )

    def test_duplicates_within_set_and_against_trashnet_are_flagged(self):
        source = self.photos / "glass/glass0.png"
        (self.photos / "glass/glass_copy.png").write_bytes(source.read_bytes())
        near = np.asarray(Image.open(source)).astype(np.int16) + 3
        Image.fromarray(np.clip(near, 0, 255).astype(np.uint8)).save(self.photos / "metal/zz.png")
        trash_phash = str(imagehash.phash(Image.open(self.photos / "paper/paper1.png")))
        self._write_trashnet_records(
            [
                {
                    "image_id": "paper/paper999.jpg",
                    "label": "paper",
                    "sha256": "f" * 64,
                    "phash": trash_phash,
                    "included": True,
                    "duplicate_group": "paper/paper999.jpg",
                    "split": "test",
                }
            ]
        )
        output = self.audit()
        manifest = self.manifest().set_index("image_id")
        self.assertEqual(manifest.loc["glass/glass_copy.png", "exclusion_reason"], rob.EXACT_WITHIN)
        self.assertEqual(manifest.loc["metal/zz.png", "exclusion_reason"], rob.NEAR_WITHIN)
        self.assertEqual(manifest.loc["paper/paper1.png", "exclusion_reason"], rob.NEAR_TRASHNET)
        self.assertEqual(manifest.loc["glass/glass0.png", "included"], "True")
        flags = pd.read_csv(self.root / "docs/robustness/duplicate-flags.csv")
        self.assertEqual(
            set(flags["kind"]), {rob.EXACT_WITHIN, rob.NEAR_WITHIN, rob.NEAR_TRASHNET}
        )
        self.assertIn("duplicate flags", output)

    def test_gps_exif_is_warned_and_excluded(self):
        image = _noise(self.rng)
        exif = Image.Exif()
        gps = exif.get_ifd(rob.GPS_IFD_TAG)
        gps[1] = "N"
        gps[2] = (51.0, 30.0, 0.0)
        image.save(self.photos / "metal/located.jpg", exif=exif)
        _noise(self.rng).save(self.photos / "metal/clean.jpg", exif=Image.Exif())
        output = self.audit()
        manifest = self.manifest().set_index("image_id")
        self.assertEqual(manifest.loc["metal/located.jpg", "exclusion_reason"], rob.GPS_EXIF)
        self.assertEqual(manifest.loc["metal/clean.jpg", "included"], "True")
        self.assertIn("GPS", output)
        self.assertIn("metal/located.jpg", output)
        summary = json.loads((self.root / "docs/robustness/audit-summary.json").read_text())
        self.assertEqual(summary["gps_warnings"], ["metal/located.jpg"])

    def test_unreadable_and_misplaced_files(self):
        (self.photos / "plastic/broken.jpg").write_bytes(b"not an image")
        self.audit()
        manifest = self.manifest().set_index("image_id")
        self.assertEqual(manifest.loc["plastic/broken.jpg", "exclusion_reason"], rob.UNREADABLE)
        (self.photos / "wood").mkdir()
        _noise(self.rng).save(self.photos / "wood/x.png")
        with self.assertRaisesRegex(rob.RobustnessError, "Unexpected file"):
            self.audit()


class IsolationTests(RobustnessFixture):
    def test_nothing_from_the_robustness_set_can_enter_training(self):
        self.audit()
        # 1. The robustness manifest has no split column, so the training loader rejects it.
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_manifest(self.root / "docs/robustness/manifest.csv")
        # 2. Training reads only the TrashNet manifest and image root, never these paths.
        config = tomllib.loads((REPO / "configs/phase2.toml").read_text())
        self.assertEqual(config["paths"]["image_root"], "data/raw/trashnet")
        self.assertEqual(config["paths"]["manifest"], "docs/dataset/manifest.csv")
        training_ids = set(load_manifest(self.root / "docs/dataset/manifest.csv")["image_id"])
        self.assertFalse(training_ids & set(self.manifest()["image_id"]))
        # 3. Photos may not live inside the TrashNet image root.
        inside = self.root / "data/raw/trashnet/robustness"
        (inside / "glass").mkdir(parents=True)
        _noise(self.rng).save(inside / "glass/a.png")
        with self.assertRaisesRegex(rob.RobustnessError, "outside the TrashNet"):
            rob.run_robustness_audit(self.root, image_root="data/raw/trashnet/robustness")
        # 4. A byte-identical TrashNet image is excluded.
        photo = self.photos / "cardboard/cardboard0.png"
        self._write_trashnet_records(
            [
                {
                    "image_id": "cardboard/cardboard500.jpg",
                    "label": "cardboard",
                    "sha256": sha256(photo),
                    "phash": "0" * 16,
                    "included": True,
                    "duplicate_group": "cardboard/cardboard500.jpg",
                    "split": "train",
                }
            ]
        )
        self.audit()
        manifest = self.manifest().set_index("image_id")
        self.assertEqual(
            manifest.loc["cardboard/cardboard0.png", "exclusion_reason"], rob.EXACT_TRASHNET
        )


class EvaluationTests(RobustnessFixture):
    def ready(self):
        self.audit()
        self.fill_conditions()
        self.audit()

    def evaluate(self, calls):
        return rob.run_robustness_evaluation(
            self.root, "docs/phase-2/robustness", predictor=fake_predictor(calls)
        )

    def test_single_evaluation_writes_results_and_refuses_rerun(self):
        self.ready()
        calls = []
        result = self.evaluate(calls)
        n = len(CLASSES) * PER_CLASS
        self.assertEqual(result["sample_count"], n)
        self.assertEqual(len(calls), n * len(rob.MODEL_NAMES))
        out = self.root / "docs/phase-2/robustness"
        for name in ("robustness-predictions.csv", "robustness-metrics.json", "README.md"):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((self.root / "report/robustness-section.tex").is_file())
        for model_name in rob.MODEL_NAMES:
            model = result["models"][model_name]
            self.assertEqual(set(model["per_class_recall"]), set(CLASSES))
            self.assertIn("ci_low", model["bootstrap_95_ci"]["macro_f1"])
            self.assertAlmostEqual(
                model["change_vs_trashnet"]["accuracy"], model["accuracy"] - 0.93
            )
            for field in rob.CONDITION_FIELDS:
                counts = sum(v["n"] for v in result["by_condition"][field].values())
                self.assertEqual(counts, n)
        self.assertEqual(len(result["mcnemar"]), 3)

        with self.assertRaises(FileExistsError):
            self.evaluate(calls)
        self.assertEqual(len(calls), n * len(rob.MODEL_NAMES), "rerun must not score anything")
        # Rebuilding the documents from the saved file scores nothing either.
        before = (out / "robustness-predictions.csv").read_bytes()
        rob.evaluate_robustness_from_predictions(self.root, "docs/phase-2/robustness")
        self.assertEqual((out / "robustness-predictions.csv").read_bytes(), before)

    def test_protocol_json_is_never_written(self):
        before_bytes = self.protocol_path.read_bytes()
        before_mtime = self.protocol_path.stat().st_mtime_ns
        mode = self.protocol_path.stat().st_mode
        os.chmod(self.protocol_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        self.addCleanup(os.chmod, self.protocol_path, mode)
        self.ready()
        self.evaluate([])
        rob.evaluate_robustness_from_predictions(self.root, "docs/phase-2/robustness")
        self.assertEqual(self.protocol_path.read_bytes(), before_bytes)
        self.assertEqual(self.protocol_path.stat().st_mtime_ns, before_mtime)

    def test_refuses_to_write_into_the_frozen_evaluation_folder(self):
        self.ready()
        with self.assertRaisesRegex(rob.RobustnessError, "frozen protocol"):
            rob.run_robustness_evaluation(
                self.root, "docs/phase-2/evaluation", predictor=fake_predictor([])
            )

    def test_refuses_incomplete_conditions_changed_photos_and_edited_manifest(self):
        self.audit()
        with self.assertRaisesRegex(rob.RobustnessError, "must be filled in"):
            self.evaluate([])
        self.fill_conditions()
        self.audit()
        photo = self.photos / "trash/trash0.png"
        _noise(self.rng).save(photo)
        with self.assertRaisesRegex(rob.RobustnessError, "changed since the audit"):
            self.evaluate([])
        self.audit()
        manifest_path = self.root / "docs/robustness/manifest.csv"
        manifest_path.write_text(manifest_path.read_text().replace("True", "False", 1))
        with self.assertRaisesRegex(rob.RobustnessError, "changed since the audit"):
            self.evaluate([])
        self.assertFalse((self.root / "docs/phase-2/robustness").exists())

    def test_threshold_mismatch_is_refused(self):
        self.ready()

        def wrong(model_name, image, root="."):
            result = fake_predictor([])(model_name, image, root=root)
            result["threshold"] = 0.1
            return result

        with self.assertRaisesRegex(rob.RobustnessError, "freezes"):
            rob.run_robustness_evaluation(self.root, predictor=wrong)


class DispatcherTests(unittest.TestCase):
    def test_other_commands_are_delegated_to_phase2_evaluation(self):
        with patch("waste_sorting.phase2_evaluation.main") as delegated:
            rob.main(["test", "--output", "x"])
        delegated.assert_called_once_with(["test", "--output", "x"])

    def test_entry_point_targets_the_dispatcher(self):
        project = tomllib.loads((REPO / "pyproject.toml").read_text())
        self.assertEqual(
            project["project"]["scripts"]["waste-phase2-eval"],
            "waste_sorting.robustness_evaluation:main",
        )


if __name__ == "__main__":
    unittest.main()
