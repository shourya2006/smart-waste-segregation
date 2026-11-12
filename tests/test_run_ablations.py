"""Tests for scripts/run_ablations.py: configs, resume logic, dry-run, summarise."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_ablations import (
    ABLATIONS,
    handle_run_directory,
    run_ablations,
    summarise_ablations,
)


class RunAblationsTests(unittest.TestCase):
    def test_ablation_configs_and_overrides_match_plan(self):
        """Verify the 13 configurations match docs/phase-2/ablation-plan.md exactly."""
        self.assertEqual(len(ABLATIONS), 13)

        expected = [
            ("BASE", "Baseline", ()),
            ("A1.1", "Architecture", ("dl.training.backbone=efficientnet_b0",)),
            ("A2.1", "Fine-Tuning", ("dl.training.unfreeze_blocks=2",)),
            ("A3.1", "Resolution", ("preprocessing.width=160", "preprocessing.height=160")),
            ("A4.0", "Dropout", ("dl.training.dropout=0.0",)),
            ("A4.2", "Dropout", ("dl.training.dropout=0.5",)),
            ("A5.1", "Augmentation", ("dl.augmentation.enabled=false",)),
            ("A5.2", "Augmentation", ("dl.augmentation.random_resized_crop=false",)),
            ("A5.3", "Augmentation", ("dl.augmentation.horizontal_flip=false",)),
            ("A5.4", "Augmentation", ("dl.augmentation.rotation=false",)),
            ("A5.5", "Augmentation", ("dl.augmentation.color_jitter=false",)),
            ("A6.1", "Imbalance", ("dl.training.class_weighted_loss=true",)),
            ("A6.2", "Imbalance", ("dl.data.balanced_sampling=true",)),
        ]

        for i, (exp_id, exp_factor, exp_overrides) in enumerate(expected):
            cfg = ABLATIONS[i]
            self.assertEqual(cfg.id, exp_id)
            self.assertEqual(cfg.factor, exp_factor)
            self.assertEqual(cfg.overrides, exp_overrides)

    def test_dry_run_prints_thirteen_commands(self):
        """--dry-run prints exactly 13 commands each with one change from BASE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_ablations(runs_dir=Path(tmpdir) / "runs", dry_run=True)
            lines = [line for line in buf.getvalue().splitlines() if "(dry-run)" in line]
            self.assertEqual(len(lines), 13)
            self.assertIn(f"waste-dl-train cv --output {Path(tmpdir) / 'runs' / 'BASE'}", lines[0])
            self.assertIn("--set dl.training.backbone=efficientnet_b0", lines[1])

    def test_handle_run_directory_skip_completed(self):
        """Completed run with summary.json is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "BASE"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text("{}")
            is_done = handle_run_directory(run_dir)
            self.assertTrue(is_done)
            self.assertTrue(run_dir.exists())

    def test_handle_run_directory_renames_interrupted(self):
        """Interrupted run without summary.json is renamed to .interrupted-<timestamp>."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "BASE"
            run_dir.mkdir()
            (run_dir / "runs.csv").write_text("fold,macro_f1\n0,0.85\n")
            is_done = handle_run_directory(run_dir)
            self.assertFalse(is_done)
            self.assertFalse(run_dir.exists())
            # A renamed directory should exist
            interrupted = list(Path(tmpdir).glob("BASE.interrupted-*"))
            self.assertEqual(len(interrupted), 1)

    def test_run_ablations_only_subset(self):
        """--only flag selects specific IDs and rejects invalid ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_ablations(
                    runs_dir=Path(tmpdir) / "runs",
                    only_ids=["BASE", "A1.1"],
                    dry_run=True,
                )
            lines = [line for line in buf.getvalue().splitlines() if "(dry-run)" in line]
            self.assertEqual(len(lines), 2)
            self.assertIn("BASE", lines[0])
            self.assertIn("A1.1", lines[1])

            with self.assertRaises(ValueError):
                run_ablations(runs_dir=Path(tmpdir) / "runs", only_ids=["NONEXISTENT"])

    def test_summarise_builds_correct_table(self):
        """`summarise` builds the correct table from fake runs.csv and summary.json."""
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()

            # Create mock runs for BASE and A1.1
            for ab_id, f1, acc, params in [("BASE", 0.88, 0.89, 7686), ("A1.1", 0.87, 0.88, 7686)]:
                d = runs_dir / ab_id
                d.mkdir()
                runs_rows = []
                for fold in range(5):
                    row = {
                        "fold": fold,
                        "macro_f1": f1 + 0.01 * fold,
                        "accuracy": acc,
                        "epochs_run": 20,
                        "train_seconds": 150.0,
                    }
                    for c in classes:
                        row[f"recall_{c}"] = 0.85
                    runs_rows.append(row)
                pd.DataFrame(runs_rows).to_csv(d / "runs.csv", index=False)

                summary = {
                    "mode": "cv",
                    "mean_macro_f1": f1 + 0.02,
                    "std_macro_f1": 0.015,
                    "mean_accuracy": acc,
                    "median_best_epoch": 18,
                    "device": "cuda",
                    "trainable_parameters": params,
                    "config": {"classes": classes},
                }
                (d / "summary.json").write_text(json.dumps(summary))

            out_csv = Path(tmpdir) / "summary.csv"
            df = summarise_ablations(runs_dir, output_csv=out_csv)

            self.assertEqual(len(df), 2)
            self.assertEqual(list(df["id"]), ["BASE", "A1.1"])
            self.assertIn("mean_macro_f1", df.columns)
            self.assertIn("std_macro_f1", df.columns)
            self.assertIn("mean_accuracy", df.columns)
            for c in classes:
                self.assertIn(f"mean_recall_{c}", df.columns)
            self.assertIn("mean_epochs_run", df.columns)
            self.assertIn("mean_train_seconds", df.columns)
            self.assertIn("device", df.columns)
            self.assertIn("trainable_parameters", df.columns)

            self.assertAlmostEqual(df.loc[df["id"] == "BASE", "mean_macro_f1"].iloc[0], 0.90)
            self.assertEqual(df.loc[df["id"] == "BASE", "device"].iloc[0], "cuda")
            self.assertEqual(df.loc[df["id"] == "BASE", "median_best_epoch"].iloc[0], 18)
            self.assertTrue(out_csv.is_file())

    def test_summarise_raises_if_median_best_epoch_missing(self):
        """`summarise` raises KeyError naming the run if median_best_epoch is absent."""
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()
            d = runs_dir / "BASE"
            d.mkdir()
            (d / "runs.csv").write_text("fold,macro_f1,epochs_run,train_seconds\n0,0.85,20,100\n")
            summary = {
                "mode": "cv",
                "mean_macro_f1": 0.85,
                "std_macro_f1": 0.01,
                "mean_accuracy": 0.86,
                # omitted "median_best_epoch"
                "device": "mps",
                "trainable_parameters": 7686,
                "config": {"classes": classes},
            }
            (d / "summary.json").write_text(json.dumps(summary))

            with self.assertRaises(KeyError) as ctx:
                summarise_ablations(runs_dir)
            self.assertIn("BASE", str(ctx.exception))
            self.assertIn("median_best_epoch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
