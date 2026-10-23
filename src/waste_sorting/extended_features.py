"""Opt-in, image-local colour/texture descriptors for the SVM follow-up.

These do not replace the shared 1,806-dimensional baseline feature contract.
All histogram bins, regions and foreground rules are fixed before fitting.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import local_binary_pattern

from waste_sorting.preprocessing import preprocess_image

SETTINGS = {
    "preprocessing": {
        "width": 256,
        "height": 256,
        "preserve_aspect_ratio": True,
        "padding_rgb": [255, 255, 255],
    },
    "hsv_bins": [32, 16, 16],
    "rgb_bins": 16,
    "lab_bins": 16,
    "joint_hs_bins": [16, 8],
    "lbp_uniform": [[8, 1], [16, 2], [24, 3]],
    "lbp_nonuniform_rotation_sensitive": [[8, 1], [8, 2]],
    "foreground_rule": "HSV saturation > 0.12 OR value < 0.92",
    "regions": ["global", "foreground", "top_left", "top_right", "bottom_left", "bottom_right"],
}
COLOR_DIM = 32 + 16 + 16 + 3 * 16 + 3 * 16 + 16 * 8
TEXTURE_DIM = 10 + 18 + 26 + 59 + 59
REGION_DIM = COLOR_DIM + TEXTURE_DIM
DIMENSION = 6 * REGION_DIM


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return values / max(float(values.sum()), 1.0)


def extract_extended_features(image: Image.Image) -> np.ndarray:
    """Return fixed-order float32 histograms; no dataset statistics or labels used."""
    rgb = np.asarray(preprocess_image(image, SETTINGS), dtype=np.float64) / 255.0
    hsv, lab = rgb2hsv(rgb), rgb2lab(rgb)
    gray = np.rint(rgb2gray(rgb) * 255).astype(np.uint8)
    textures = [
        (local_binary_pattern(gray, p, r, "uniform"), p + 2) for p, r in SETTINGS["lbp_uniform"]
    ] + [
        (local_binary_pattern(gray, p, r, "nri_uniform"), 59)
        for p, r in SETTINGS["lbp_nonuniform_rotation_sensitive"]
    ]
    h, w = gray.shape
    full = np.ones((h, w), dtype=bool)
    masks = [full, (hsv[:, :, 1] > 0.12) | (hsv[:, :, 2] < 0.92)]
    for row in range(2):
        for col in range(2):
            mask = np.zeros_like(full)
            mask[row * h // 2 : (row + 1) * h // 2, col * w // 2 : (col + 1) * w // 2] = True
            masks.append(mask)
    parts = []
    for mask in masks:
        color = []
        for space, bins, ranges in [
            (hsv, SETTINGS["hsv_bins"], [(0, 1)] * 3),
            (rgb, [16] * 3, [(0, 1)] * 3),
            (lab, [16] * 3, [(0, 100), (-128, 128), (-128, 128)]),
        ]:
            for ch, (n, bounds) in enumerate(zip(bins, ranges, strict=True)):
                color.append(_distribution(np.histogram(space[:, :, ch][mask], n, bounds)[0]))
        color.append(
            _distribution(
                np.histogram2d(
                    hsv[:, :, 0][mask], hsv[:, :, 1][mask], bins=[16, 8], range=[(0, 1), (0, 1)]
                )[0]
            )
        )
        # Equal family mass prevents the number of channels/bins setting its weight.
        parts.append(_distribution(np.concatenate(color)).astype(np.float32))
        texture_mask = mask.copy()
        texture_mask[:3] = texture_mask[-3:] = False
        texture_mask[:, :3] = texture_mask[:, -3:] = False
        texture = [
            _distribution(np.bincount(t[texture_mask].astype(int), minlength=n))
            for t, n in textures
        ]
        parts.append(_distribution(np.concatenate(texture)).astype(np.float32))
    result = np.concatenate(parts)
    if result.shape != (DIMENSION,) or not np.isfinite(result).all():
        raise ValueError("Invalid extended feature vector")
    return result


def extended_columns(family):
    """Four predeclared ablations, with canonical full-vector input at inference."""
    definitions = {
        "global_color": (range(2), []),
        "global_color_texture": (range(2), range(2)),
        "global_color_spatial_texture": (range(2), range(6)),
        "spatial_color_texture": (range(6), range(6)),
    }
    if family not in definitions:
        raise ValueError(f"Unknown extended feature family: {family}")
    color_regions, texture_regions = definitions[family]
    columns = []
    for region in range(6):
        start = region * REGION_DIM
        if region in color_regions:
            columns.extend(range(start, start + COLOR_DIM))
        if region in texture_regions:
            columns.extend(range(start + COLOR_DIM, start + REGION_DIM))
    return columns
