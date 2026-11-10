"""Unit tests for pretrained backbones and parameter freezing."""

import unittest

try:
    import torch
    import torch.nn as nn

    from waste_sorting import backbones
except ImportError:
    torch = None

SKIP_REASON = "PyTorch is not installed (pip install -r requirements-dl.lock)"


@unittest.skipIf(torch is None, SKIP_REASON)
class BackboneTests(unittest.TestCase):
    def test_load_backbone_unsupported_name_raises(self):
        with self.assertRaises(ValueError):
            backbones.load_backbone("resnet50", pretrained=False)

    def test_output_shape_mobilenet_v2_224_and_160(self):
        model, dim = backbones.load_backbone("mobilenet_v2", pretrained=False)
        self.assertEqual(dim, 1280)
        self.assertIsInstance(model.classifier, nn.Identity)

        # Test at 224x224
        x224 = torch.randn(2, 3, 224, 224)
        out224 = model(x224)
        self.assertEqual(out224.shape, (2, 1280))

        # Test at 160x160
        x160 = torch.randn(2, 3, 160, 160)
        out160 = model(x160)
        self.assertEqual(out160.shape, (2, 1280))

    def test_output_shape_efficientnet_b0_224_and_160(self):
        model, dim = backbones.load_backbone("efficientnet_b0", pretrained=False)
        self.assertEqual(dim, 1280)
        self.assertIsInstance(model.classifier, nn.Identity)

        # Test at 224x224
        x224 = torch.randn(2, 3, 224, 224)
        out224 = model(x224)
        self.assertEqual(out224.shape, (2, 1280))

        # Test at 160x160
        x160 = torch.randn(2, 3, 160, 160)
        out160 = model(x160)
        self.assertEqual(out160.shape, (2, 1280))

    def test_freeze_leaves_no_trainable_parameters(self):
        for name in ("mobilenet_v2", "efficientnet_b0"):
            with self.subTest(backbone=name):
                model, _ = backbones.load_backbone(name, pretrained=False)
                # Initially some params are trainable
                self.assertTrue(any(p.requires_grad for p in model.parameters()))
                backbones.freeze(model)
                # Now none are trainable
                self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_unfreeze_last_blocks_mobilenet_v2(self):
        model, _ = backbones.load_backbone("mobilenet_v2", pretrained=False)
        backbones.freeze(model)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

        # Unfreeze last 2 blocks (index 17: InvertedResidual, index 18: Conv2dNormActivation)
        backbones.unfreeze_last_blocks(model, 2)
        total_blocks = len(model.features)  # 19

        for i in range(total_blocks - 2):
            self.assertFalse(
                any(p.requires_grad for p in model.features[i].parameters()),
                f"Block {i} should remain frozen",
            )
        for i in range(total_blocks - 2, total_blocks):
            self.assertTrue(
                all(p.requires_grad for p in model.features[i].parameters()),
                f"Block {i} should be unfrozen",
            )

    def test_unfreeze_last_blocks_efficientnet_b0(self):
        model, _ = backbones.load_backbone("efficientnet_b0", pretrained=False)
        backbones.freeze(model)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

        # Unfreeze last 3 stages
        backbones.unfreeze_last_blocks(model, 3)
        total_blocks = len(model.features)  # 9

        for i in range(total_blocks - 3):
            self.assertFalse(
                any(p.requires_grad for p in model.features[i].parameters()),
                f"Stage {i} should remain frozen",
            )
        for i in range(total_blocks - 3, total_blocks):
            self.assertTrue(
                all(p.requires_grad for p in model.features[i].parameters()),
                f"Stage {i} should be unfrozen",
            )

    def test_unfreeze_zero_or_negative_does_nothing(self):
        model, _ = backbones.load_backbone("mobilenet_v2", pretrained=False)
        backbones.freeze(model)
        backbones.unfreeze_last_blocks(model, 0)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        backbones.unfreeze_last_blocks(model, -2)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_trainable_parameter_counts_for_unfreeze_blocks(self):
        # MobileNetV2 total backbone: 2,223,872
        # n=1 (last 1x1 conv): 412,160
        # n=2 (+ last inverted residual): 886,080
        # n=3 (+ second-to-last inverted residual): 1,206,080
        mnet_counts = {1: 412_160, 2: 886_080, 3: 1_206_080}
        for n, expected_count in mnet_counts.items():
            model, _ = backbones.load_backbone("mobilenet_v2", pretrained=False)
            backbones.freeze(model)
            backbones.unfreeze_last_blocks(model, n)
            self.assertEqual(backbones.count_trainable_parameters(model), expected_count)

        # EfficientNet-B0 total backbone: 4,007,548
        # n=1 (stage 8 conv): 412,160
        # n=2 (+ stage 7 MBConv): 1,129,392
        # n=3 (+ stage 6 MBConvs): 3,155,740
        effnet_counts = {1: 412_160, 2: 1_129_392, 3: 3_155_740}
        for n, expected_count in effnet_counts.items():
            model, _ = backbones.load_backbone("efficientnet_b0", pretrained=False)
            backbones.freeze(model)
            backbones.unfreeze_last_blocks(model, n)
            self.assertEqual(backbones.count_trainable_parameters(model), expected_count)

    def test_keep_frozen_bn_eval_preserves_running_stats(self):
        model, _ = backbones.load_backbone("mobilenet_v2", pretrained=False)
        backbones.freeze(model)
        # Unfreeze only the final block (index 18)
        backbones.unfreeze_last_blocks(model, 1)

        # Calling model.train() switches all modules including frozen BNs to train mode
        model.train()
        # Helper must set only frozen BNs back to eval mode
        backbones.keep_frozen_bn_eval(model)

        frozen_bn = model.features[0][1]
        unfrozen_bn = model.features[18][1]

        self.assertFalse(frozen_bn.training)
        self.assertTrue(unfrozen_bn.training)

        init_frozen_mean = frozen_bn.running_mean.clone()
        init_unfrozen_mean = unfrozen_bn.running_mean.clone()

        # Run forward pass with synthetic inputs
        x = torch.randn(4, 3, 224, 224)
        model(x)

        # Frozen BN running statistics must be strictly unchanged
        self.assertTrue(torch.equal(frozen_bn.running_mean, init_frozen_mean))
        # Unfrozen BN running statistics must have been updated
        self.assertFalse(torch.equal(unfrozen_bn.running_mean, init_unfrozen_mean))


if __name__ == "__main__":
    unittest.main()
