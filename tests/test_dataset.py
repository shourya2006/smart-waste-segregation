"""Behavioral checks for integrity auditing, duplicate decisions, and split isolation."""

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting.acquisition import CLASSES, acquire, extract_archive
from waste_sorting.dataset import (
    apply_reviews,
    assign_splits,
    audit_images,
    duplicate_candidates,
)


class ArchiveTests(unittest.TestCase):
    def test_zip_slip_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("../escaped.jpg", b"not an image")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_archive(archive, root / "images")
            self.assertFalse((root / "escaped.jpg").exists())

    def test_wrong_archive_checksum_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "wrong.zip"
            archive.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "checksum"):
                acquire(root, archive)
            self.assertFalse((root / "data/raw/trashnet").exists())

    def test_existing_image_edits_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("dataset-resized/paper/a.jpg", b"original")
            extract_archive(archive, root / "images")
            extract_archive(archive, root / "images")
            (root / "images/paper/a.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "modified"):
                extract_archive(archive, root / "images")
            self.assertEqual((root / "images/paper/a.jpg").read_bytes(), b"changed")


class AuditTests(unittest.TestCase):
    def test_corrupt_files_and_duplicate_content_are_accounted_for(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "paper").mkdir()
            pixels = np.random.default_rng(1).integers(0, 255, (24, 24, 3), np.uint8)
            image = Image.fromarray(pixels)
            image.save(root / "paper/a.png")
            # A different encoding of identical pixels must not evade duplicate checks.
            image.save(root / "paper/b.bmp")
            shutil.copyfile(root / "paper/a.png", root / "paper/c.png")
            (root / "paper/broken.jpg").write_bytes(b"broken")
            frame = audit_images(root)
            self.assertEqual(len(frame), 4)
            self.assertEqual(int(frame.included.sum()), 1)
            self.assertEqual(int(frame.exclusion_reason.eq("duplicate_content").sum()), 2)
            self.assertEqual(int(frame.read_status.eq("unreadable").sum()), 1)
            self.assertEqual(frame.loc[frame.included, "canonical_id"].item(), "paper/a.png")

    def test_conflicting_labels_quarantine_all_identical_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ("paper", "metal"):
                (root / label).mkdir()
                Image.new("RGB", (16, 16), "white").save(root / label / "a.png")
            frame = audit_images(root)
            self.assertFalse(frame.included.any())
            self.assertTrue(frame.exclusion_reason.eq("conflicting_labels").all())

    def test_unknown_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unknown").mkdir()
            (root / "unknown/a.png").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                audit_images(root)


class ReviewAndSplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for label in CLASSES:
            for index in range(40):
                identifier = f"{label}/{index:02d}.jpg"
                rows.append(
                    {
                        "image_id": identifier,
                        "label": label,
                        "included": True,
                        "read_status": "ok",
                        "canonical_id": identifier,
                        "duplicate_group": identifier,
                        "sha256": identifier,
                        "pixel_sha256": identifier,
                    }
                )
        self.frame = pd.DataFrame(rows)
        self.fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
        self.candidates = pd.DataFrame(
            [
                {
                    "pair_id": "pair",
                    "left_id": "paper/00.jpg",
                    "right_id": "paper/01.jpg",
                }
            ]
        )

    def reviews(self, decision):
        return pd.DataFrame(
            [
                {
                    "pair_id": "pair",
                    "decision": decision,
                    "reviewer": "test reviewer",
                    "reason": "explicit fixture decision",
                }
            ]
        )

    def test_pending_review_blocks_splitting(self):
        with self.assertRaisesRegex(ValueError, "Every candidate"):
            apply_reviews(self.frame, self.candidates, self.reviews(""))

    def test_stale_review_ids_are_rejected(self):
        reviews = self.reviews("same_item")
        reviews.loc[0, "pair_id"] = "old pair"
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_reviews(self.frame, self.candidates, reviews)

    def test_same_and_uncertain_pairs_stay_in_same_split(self):
        for decision in ("same_item", "uncertain"):
            grouped = apply_reviews(self.frame, self.candidates, self.reviews(decision))
            split = assign_splits(grouped, self.fractions, 42).set_index("image_id")
            self.assertEqual(split.loc["paper/00.jpg", "split"], split.loc["paper/01.jpg", "split"])

    def test_different_items_are_not_merged(self):
        grouped = apply_reviews(self.frame, self.candidates, self.reviews("different_items"))
        self.assertEqual(grouped.duplicate_group.nunique(), len(self.frame))

    def test_split_is_reproducible_and_stratified(self):
        first = assign_splits(self.frame, self.fractions, 42)
        second = assign_splits(self.frame, self.fractions, 42)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.split.equals(assign_splits(self.frame, self.fractions, 99).split))
        counts = pd.crosstab(first.label, first.split)
        for label in CLASSES:
            self.assertEqual(counts.loc[label, "train"], 28)
            self.assertEqual(counts.loc[label, "validation"], 6)
            self.assertEqual(counts.loc[label, "test"], 6)

    def test_excluded_duplicate_is_never_a_training_example(self):
        frame = self.frame.copy()
        frame.loc[0, "included"] = False
        result = assign_splits(frame, self.fractions, 42)
        self.assertEqual(result.loc[0, "split"], "excluded")

    def test_invalid_ratios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            assign_splits(self.frame, {"train": 0.8, "validation": 0.2, "test": 0.2}, 42)

    def test_missing_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "All six"):
            assign_splits(self.frame[self.frame.label.ne("trash")], self.fractions, 42)

    def test_perceptual_candidates_use_inclusive_hamming_distance(self):
        frame = pd.DataFrame(
            [
                {"image_id": "paper/a", "label": "paper", "included": True, "phash": "0000"},
                {"image_id": "paper/b", "label": "paper", "included": True, "phash": "0003"},
                {"image_id": "paper/c", "label": "paper", "included": False, "phash": "0000"},
            ]
        )
        self.assertEqual(len(duplicate_candidates(frame, 1)), 0)
        self.assertEqual(len(duplicate_candidates(frame, 2)), 1)


if __name__ == "__main__":
    unittest.main()
