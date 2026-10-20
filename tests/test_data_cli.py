"""Exercise the audit-to-split commands on an independent synthetic dataset."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting.acquisition import CLASSES
from waste_sorting.data_cli import main


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/phase1.toml").write_text(
            "seed = 42\n[splits]\ntrain = 0.7\nvalidation = 0.15\ntest = 0.15\n"
        )
        rng = np.random.default_rng(14)
        for label in CLASSES:
            folder = self.root / "data/raw/trashnet" / label
            folder.mkdir(parents=True)
            for index in range(12):
                pixels = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
        self.command("audit", "--phash-distance", "0")

    def command(self, *arguments):
        with patch("sys.argv", ["waste-data", "--root", str(self.root), *arguments]):
            with contextlib.redirect_stdout(io.StringIO()):
                main()

    def test_commands_preserve_every_usable_image_and_reproduce_manifest(self):
        self.command("split")
        path = self.root / "docs/dataset/manifest.csv"
        first = path.read_bytes()
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 72)
        self.assertTrue(frame.included.all())
        self.assertEqual(set(frame.split), {"train", "validation", "test"})
        self.assertEqual(frame.groupby("duplicate_group").split.nunique().max(), 1)
        self.command("split")
        self.assertEqual(path.read_bytes(), first)
        self.command("eda")
        self.assertTrue((self.root / "docs/dataset/figures/train-class-balance.png").exists())

    def test_changed_image_invalidates_audit(self):
        image = self.root / "data/raw/trashnet/paper/0.png"
        Image.new("RGB", (24, 24), "red").save(image)
        with self.assertRaisesRegex(ValueError, "changed since audit"):
            self.command("split")

    def test_new_image_invalidates_audit(self):
        Image.new("RGB", (24, 24), "red").save(self.root / "data/raw/trashnet/paper/new.png")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.command("split")

    def test_edited_audit_output_is_rejected(self):
        audit = self.root / "docs/dataset/audit_manifest.csv"
        audit.write_text(audit.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Audit output changed"):
            self.command("split")


if __name__ == "__main__":
    unittest.main()
