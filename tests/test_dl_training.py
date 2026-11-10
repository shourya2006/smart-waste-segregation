"""Checks for the Phase 2 training loop on tiny synthetic data (CPU, random weights)."""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch

    from waste_sorting import dl_data, dl_training
except ImportError:  # torch/torchvision are the optional "dl" extra
    torch = None

REPO = Path(__file__).resolve().parents[1]
SKIP_REASON = "PyTorch is not installed (pip install -r requirements-dl.lock)"
HEAD_PARAMETERS = 1280 * 6 + 6


@unittest.skipIf(torch is None, SKIP_REASON)
class DlTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = tomllib.loads((REPO / "configs/phase2.toml").read_text())
        config["preprocessing"].update(width=64, height=64)
        config["dl"]["data"].update(batch_size=8, num_workers=0)
        config["dl"]["training"].update(max_epochs=2, patience=1)
        self.config = config
        self.classes = config["classes"]
        rows, rng = [], np.random.default_rng(0)
        for label in self.classes:
            for split, count in (("train", 6), ("validation", 1), ("test", 1)):
                for n in range(count):
                    image_id = f"{label}/{split}-{n}.png"
                    rows.append({"image_id": image_id, "label": label, "included": True,
                                 "duplicate_group": image_id, "split": split})
                    path = self.root / "data/raw/trashnet" / image_id
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(path)
        manifest = pd.DataFrame(rows)
        (self.root / "docs/dataset").mkdir(parents=True)
        manifest.to_csv(self.root / "docs/dataset/manifest.csv", index=False)
        self.train_ids = sorted(manifest.loc[manifest["split"] == "train", "image_id"])
        folds = {image_id: i % 3 for i, image_id in enumerate(self.train_ids)}
        (self.root / "docs/training").mkdir(parents=True)
        (self.root / "docs/training/folds.json").write_text(
            json.dumps({"n_splits": 3, "assignments": folds})
        )
        patcher = mock.patch.object(dl_training, "pick_device", return_value=torch.device("cpu"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cpu = torch.device("cpu")

    def configured(self, *overrides):
        return dl_training.apply_overrides(self.config, overrides)

    def model(self, *overrides):
        torch.manual_seed(0)
        return dl_training.build_model(self.configured(*overrides), pretrained=False)

    def loaders(self, config=None, fold=0):
        torch.manual_seed(0)
        return dl_data.make_dataloaders(config or self.config, self.root, fold=fold)

    def test_frozen_model_trains_only_the_head(self):
        model = self.model()
        self.assertEqual(dl_training.count_trainable_parameters(model), HEAD_PARAMETERS)
        unfrozen = self.model("dl.training.unfreeze_blocks=2")
        # 886,080 = MobileNetV2 features[17:19], as listed in the ablation plan.
        self.assertEqual(
            dl_training.count_trainable_parameters(unfrozen), 886_080 + HEAD_PARAMETERS
        )

    def test_optimizer_uses_lower_learning_rate_for_unfrozen_blocks(self):
        settings = self.config["dl"]["training"]
        frozen = dl_training.make_optimizer(self.model(), settings)
        self.assertEqual([g["lr"] for g in frozen.param_groups], [settings["lr_head"]])
        partial = dl_training.make_optimizer(
            self.model("dl.training.unfreeze_blocks=1"), settings
        )
        self.assertEqual(
            [g["lr"] for g in partial.param_groups],
            [settings["lr_head"], settings["lr_backbone"]],
        )

    def test_frozen_backbone_weights_and_statistics_do_not_change(self):
        config = self.configured('dl.training.backbone="efficientnet_b0"')
        model = self.model('dl.training.backbone="efficientnet_b0"')
        before = {k: v.clone() for k, v in model[0].state_dict().items()}
        head_before = model[1][1].weight.clone()
        loaders = self.loaders(config)
        dl_training.train_model(model, config, loaders["train"], epochs=1, device=self.cpu,
                                log=lambda _: None)
        for name, value in model[0].state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertFalse(torch.equal(model[1][1].weight, head_before))

    def test_frozen_efficientnet_is_deterministic_in_train_mode(self):
        # Stochastic depth inside frozen MBConv blocks must be off while training the head.
        model = self.model('dl.training.backbone="efficientnet_b0"')
        dl_training.set_train_mode(model)
        self.assertTrue(model[1].training)
        x = torch.randn(2, 3, 64, 64)
        self.assertTrue(torch.equal(model[0](x), model[0](x)))

    def test_unfrozen_block_batchnorm_still_updates(self):
        model = self.model("dl.training.unfreeze_blocks=1")
        first = model[0].features[0][1].running_mean.clone()
        last = model[0].features[-1][1].running_mean.clone()
        dl_training.set_train_mode(model)
        model(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(model[0].features[0][1].running_mean, first))
        self.assertFalse(torch.equal(model[0].features[-1][1].running_mean, last))

    def test_early_stopping_restores_the_best_epoch(self):
        config = self.configured("dl.training.max_epochs=4", "dl.training.patience=1",
                                 "dl.training.lr_head=0.05")
        model = self.model()
        loaders = self.loaders(config)
        result = dl_training.train_model(model, config, loaders["train"], loaders["validation"],
                                         device=self.cpu, log=lambda _: None)
        losses = [row["val_loss"] for row in result["history"]]
        self.assertEqual(result["best_epoch"], int(np.argmin(losses)) + 1)
        self.assertLessEqual(result["epochs_run"] - result["best_epoch"], 1)
        restored = dl_training.predict(model, loaders["validation"], self.cpu)[3]
        self.assertAlmostEqual(restored, min(losses), places=5)

    def test_fixed_epochs_need_a_count_and_ignore_patience(self):
        config = self.configured("dl.training.patience=1", "dl.training.max_epochs=4")
        loaders = self.loaders(config)
        with self.assertRaises(ValueError):
            dl_training.train_model(self.model(), config, loaders["train"], device=self.cpu)
        result = dl_training.train_model(self.model(), config, loaders["train"], epochs=3,
                                         device=self.cpu, log=lambda _: None)
        self.assertEqual(result["epochs_run"], 3)

    def test_final_run_follows_the_cv_learning_rate_schedule(self):
        config = self.configured("dl.training.max_epochs=4", "dl.training.patience=10")
        loaders = self.loaders(config)
        cv = dl_training.train_model(self.model(), config, loaders["train"], loaders["validation"],
                                     device=self.cpu, log=lambda _: None)
        final = dl_training.train_model(self.model(), config, loaders["train"], epochs=2,
                                        device=self.cpu, log=lambda _: None)
        self.assertEqual([r["lr"] for r in final["history"]],
                         [r["lr"] for r in cv["history"][:2]])
        self.assertLess(cv["history"][-1]["lr"], cv["history"][0]["lr"])
        with self.assertRaises(ValueError):
            dl_training.train_model(self.model(), config, loaders["train"], epochs=5,
                                    device=self.cpu)

    def test_class_weights_and_balanced_sampling_together_are_rejected(self):
        config = self.configured("dl.training.class_weighted_loss=true",
                                 "dl.data.balanced_sampling=true")
        with self.assertRaises(ValueError):
            dl_training.train_model(self.model(), config, self.loaders()["train"], epochs=1,
                                    device=self.cpu)

    def test_overrides_parse_values_and_reject_unknown_keys(self):
        config = self.configured("dl.training.dropout=0.5", "dl.augmentation.enabled=false",
                                 "dl.training.backbone=efficientnet_b0",
                                 "preprocessing.width=160")
        self.assertEqual(config["dl"]["training"]["dropout"], 0.5)
        self.assertIs(config["dl"]["augmentation"]["enabled"], False)
        self.assertEqual(config["dl"]["training"]["backbone"], "efficientnet_b0")
        self.assertEqual(config["preprocessing"]["width"], 160)
        self.assertEqual(self.config["dl"]["training"]["dropout"], 0.2)  # original untouched
        for bad in ("dl.training.droput=0.5", "dl.trainng.dropout=0.5", "dropout"):
            with self.assertRaises((KeyError, ValueError)):
                self.configured(bad)

    def test_score_reports_macro_f1_accuracy_and_recall(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.eye(6)[[0, 1, 1, 1]]
        result = dl_training.score(labels, probabilities, self.classes)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["recall_cardboard"], 0.5)
        self.assertEqual(result["recall_glass"], 1.0)
        self.assertEqual(result["recall_trash"], 0.0)

    def test_cv_scores_every_training_image_exactly_once(self):
        output = self.root / "runs/base"
        summary = dl_training.run_cv(self.config, self.root, output, pretrained=False,
                                     log=lambda _: None)
        runs = pd.read_csv(output / "runs.csv")
        self.assertEqual(list(runs["fold"]), [0, 1, 2])
        oof = pd.read_csv(output / "oof_predictions.csv")
        self.assertEqual(sorted(oof["image_id"]), self.train_ids)
        probabilities = oof[[f"prob_{c}" for c in self.classes]].sum(axis=1)
        np.testing.assert_allclose(probabilities, 1.0, atol=1e-5)
        self.assertAlmostEqual(summary["mean_macro_f1"], runs["macro_f1"].mean())
        self.assertEqual(summary["trainable_parameters"], HEAD_PARAMETERS)
        with self.assertRaises(FileExistsError):  # never overwrite a previous run
            dl_training.run_cv(self.config, self.root, output, pretrained=False)

    def test_final_model_round_trip_and_tamper_check(self):
        output = self.root / "final"
        metadata = dl_training.train_final(self.config, self.root, output, epochs=1,
                                           pretrained=False, log=lambda _: None)
        self.assertEqual(len(metadata["history"]), 1)
        self.assertNotIn("val_loss", metadata["history"][0])
        model, loaded = dl_training.load_trained_model(output)
        self.assertEqual(loaded["model_sha256"], metadata["model_sha256"])
        self.assertEqual(model(torch.randn(1, 3, 64, 64)).shape, (1, 6))
        (output / "model.pt").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            dl_training.load_trained_model(output)


if __name__ == "__main__":
    unittest.main()
