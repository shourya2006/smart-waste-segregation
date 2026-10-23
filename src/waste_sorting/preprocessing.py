"""Shared, deterministic image preparation for Phase 1 descriptors and inference."""

from collections.abc import Mapping
from numbers import Integral

from PIL import Image, ImageOps


def preprocess_image(image: Image.Image, config: Mapping) -> Image.Image:
    """Return a new EXIF-corrected RGB image using the full project config.

    Preserve aspect ratio and fit the image inside the configured canvas using
    LANCZOS. Round each resized dimension to the nearest integer (ties to even),
    clamped to [1, canvas dimension]. Center using integer floor offsets; an odd
    padding pixel goes to the right or bottom. Upscaling is allowed. Transparent
    pixels are composited onto padding_rgb before resizing. No scaling statistics,
    random augmentation, or mutation of the input image is performed.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(config, Mapping):
        raise TypeError("config must be the complete project configuration mapping")
    settings = config.get("preprocessing")
    if not isinstance(settings, Mapping):
        raise ValueError("config must contain a preprocessing mapping")

    size = []
    for key in ("width", "height"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"preprocessing.{key} must be a positive integer")
        size.append(int(value))
    width, height = size
    if settings.get("preserve_aspect_ratio", True) is not True:
        raise ValueError("Phase 1 preprocessing requires preserve_aspect_ratio=true")
    color = settings.get("padding_rgb", [255, 255, 255])
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError("padding_rgb must contain three integer channels in [0, 255]")
    if any(isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c <= 255 for c in color):
        raise ValueError("padding_rgb must contain three integer channels in [0, 255]")
    color = tuple(int(c) for c in color)
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image must have positive width and height")

    try:
        # The default non-inplace operation preserves input pixels and EXIF.
        oriented = ImageOps.exif_transpose(image)
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*color, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        scale = min(width / rgb.width, height / rgb.height)
        resized_size = (
            max(1, min(width, round(rgb.width * scale))),
            max(1, min(height, round(rgb.height * scale))),
        )
        resized = rgb.resize(resized_size, resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), color)
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError("Image pixels could not be decoded for preprocessing") from error
