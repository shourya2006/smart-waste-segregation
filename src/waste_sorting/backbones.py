"""Pretrained convolutional backbones and parameter freezing for Phase 2.

Provides MobileNetV2 and EfficientNet-B0 backbones loaded from torchvision, stripped
of their classification heads, mapping [B, 3, H, W] -> [B, 1280] (global average pooled).

Note: We use Om's letterbox + ImageNet normalisation instead of torchvision's
weights.transforms() (resize/centre-crop), so objects are not cropped; this is a
deliberate, documented choice.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

BACKBONE_WEIGHTS = {
    "mobilenet_v2": (
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.IMAGENET1K_V2,
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}

EMBEDDING_DIM = 1280


def load_backbone(
    name: str,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """Load a pretrained backbone feature extractor with classifier head removed.

    Parameters
    ----------
    name : str
        Backbone architecture name: 'mobilenet_v2' or 'efficientnet_b0'.
    pretrained : bool
        If True, loads torchvision ImageNet weights. If False, initializes with
        random weights without downloading or requiring network access (used for tests).

    Returns
    -------
    tuple[nn.Module, int]
        (feature_extractor, embedding_dim) where feature_extractor maps
        [B, 3, H, W] -> [B, 1280] (global average pooled, classifier removed),
        and embedding_dim is 1280.
    """
    if name not in BACKBONE_WEIGHTS:
        supported = sorted(BACKBONE_WEIGHTS.keys())
        raise ValueError(f"Unsupported backbone: {name!r}. Supported: {supported}")

    builder_fn, default_weights = BACKBONE_WEIGHTS[name]
    weights = default_weights if pretrained else None
    model = builder_fn(weights=weights)

    # Replace classifier with nn.Identity() so that model(x) passes through
    # features and adaptive avg pool directly into the 1280-dim flattened vector.
    # MobileNetV2: features -> AdaptiveAvgPool2d((1, 1)) -> nn.Flatten() -> classifier
    # EfficientNet-B0: features -> AdaptiveAvgPool2d((1, 1)) -> nn.Flatten() -> classifier
    model.classifier = nn.Identity()

    return model, EMBEDDING_DIM


def freeze(model: nn.Module) -> None:
    """Freeze all parameters in the model so requires_grad is False."""
    for param in model.parameters():
        param.requires_grad = False


def keep_frozen_bn_eval(model: nn.Module) -> None:
    """Ensure any BatchNorm layer whose parameters are all frozen remains in eval mode.

    When ``model.train()`` is called during fine-tuning or head training, all submodules
    are switched to training mode by default. Even when parameters have ``requires_grad=False``,
    a BatchNorm layer in training mode continues updating its ``running_mean`` and
    ``running_var`` statistics on incoming mini-batches.

    Trainers must call ``keep_frozen_bn_eval(model)`` immediately after every
    ``model.train()`` invocation to preserve pretrained ImageNet population statistics
    in frozen layers while permitting active layers to update.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_types):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the total number of trainable (requires_grad=True) parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_last_blocks(model: nn.Module, n: int) -> None:
    """Unfreeze the last n structural feature blocks of the backbone.

    Parameters
    ----------
    model : nn.Module
        MobileNetV2 or EfficientNet-B0 backbone instance.
    n : int
        Number of trailing structural blocks/stages to unfreeze.
        If n <= 0, no blocks are unfrozen (model remains fully frozen).
        If n >= total blocks, all feature blocks are unfrozen.

    Architecture specifics
    ----------------------
    - MobileNetV2:
      `model.features` contains 19 elements:
      - 0: Conv2dNormActivation (initial conv)
      - 1..17: InvertedResidual blocks (17 blocks across 7 stages)
      - 18: Conv2dNormActivation (final 1x1 expansion conv)
      Unfreezing the last n blocks means unfreezing `model.features[-n:]`.
      For instance, n=1 unfreezes the final 1x1 conv layer (index 18);
      n=2 unfreezes the last InvertedResidual block (index 17) and index 18.

    - EfficientNet-B0:
      `model.features` contains 9 stages:
      - 0: Conv2dNormActivation (initial conv3x3)
      - 1..7: Sequential stages of MBConv blocks (stage 1: 1 block, stage 2: 2 blocks,
        stage 3: 2 blocks, stage 4: 3 blocks, stage 5: 3 blocks, stage 6: 4 blocks,
        stage 7: 1 block)
      - 8: Conv2dNormActivation (final 1x1 conv)
      Unfreezing the last n blocks means unfreezing `model.features[-n:]`.
      For instance, n=1 unfreezes the final conv stage (index 8);
      n=2 unfreezes stage 7 (MBConv block) and stage 8.
    """
    if n <= 0:
        return

    if not hasattr(model, "features"):
        raise AttributeError("Model does not have a 'features' attribute.")

    total_blocks = len(model.features)
    start_idx = max(0, total_blocks - n)

    for i in range(start_idx, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True
