"""Demo helpers for the photo-upload interface.

Validates the preserved evaluated artifact, calls the shared prediction
interface, and formats accepted or abstained results. Preprocessing, feature
extraction, probability mapping, and abstention remain in the prediction path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from waste_sorting.prediction import (
    CANONICAL_CLASSES,
    PredictionResult,
    predict_image,
    validate_artifact,
)

EXPECTED_ARTIFACT_SHA256 = "83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112"
EXPECTED_ARTIFACT_SIZE = 66_270_283
DEFAULT_ARTIFACT_PATH = Path("artifacts/calibrated_svm/model.joblib")
FROZEN_ABSTENTION_THRESHOLD = 0.52
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_SIDE = 8_192
UNCERTAIN_PRIMARY = "Uncertain — manual inspection recommended"
CLASS_DISPLAY_LABELS: dict[str, str] = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Trash",
}
LIMITATIONS_NOTE = (
    "This model assigns one of six TrashNet categories: cardboard, glass, metal, "
    "paper, plastic, and trash. Calibrated confidence is not a guarantee of "
    "correctness. Confidence-based abstention does not reliably detect unknown "
    "or out-of-distribution categories. Do not treat the output as a disposal "
    "instruction; local recycling rules, contamination, and mixed materials "
    "are outside the scope of this educational demo."
)

_RUNTIME_ARTIFACT: dict[str, Any] | None = None
_RUNTIME_PATH: Path | None = None
_LOAD_COUNT = 0


class DemoError(Exception):
    """User-facing demo failure with a clear message."""


class ArtifactIntegrityError(DemoError):
    """Missing, undersized, or checksum-mismatched model file."""


@dataclass(frozen=True)
class DemoView:
    """Rendered demo fields for Gradio outputs."""

    primary_result: str
    uncertainty_message: str
    confidence_text: str
    probability_rows: list[list[str]]
    tentative_class: str
    error_message: str = ""
    used_shared_interface: bool = True


def reset_runtime_artifact() -> None:
    """Clear the process-level artifact cache. Intended for tests."""
    global _RUNTIME_ARTIFACT, _RUNTIME_PATH, _LOAD_COUNT
    _RUNTIME_ARTIFACT = None
    _RUNTIME_PATH = None
    _LOAD_COUNT = 0


def artifact_load_count() -> int:
    return _LOAD_COUNT


def empty_view(error_message: str = "") -> DemoView:
    return DemoView(
        primary_result="",
        uncertainty_message="",
        confidence_text="",
        probability_rows=[],
        tentative_class="",
        error_message=error_message,
        used_shared_interface=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_file(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_ARTIFACT_SHA256,
    expected_size: int = EXPECTED_ARTIFACT_SIZE,
) -> str:
    """Verify size and SHA-256 before the binary is unpickled."""
    if not path.exists():
        raise ArtifactIntegrityError(
            f"The evaluated model file was not found at {path}. "
            "Place the preserved artifact from Prajjwal Tripathi at this path. "
            "Do not train or download a replacement model."
        )
    size = path.stat().st_size
    if size != expected_size:
        raise ArtifactIntegrityError(
            f"Model file size is {size} bytes, which does not match the "
            f"preserved evaluated artifact ({expected_size} bytes). "
            "Refusing to load a substitute file."
        )
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ArtifactIntegrityError(
            "Model file SHA-256 does not match the preserved evaluation "
            f"artifact (expected {expected_sha256}, got {digest}). "
            "Refusing to load."
        )
    return digest


def evaluated_artifact_status(
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> str:
    """Return verified, missing, or checksum-mismatch for the saved model."""
    if not path.exists():
        return "missing"
    try:
        verify_artifact_file(path)
    except ArtifactIntegrityError:
        return "checksum-mismatch"
    return "verified"


def load_evaluated_artifact(
    path: Path | str | None = None,
    *,
    expected_sha256: str = EXPECTED_ARTIFACT_SHA256,
    expected_size: int = EXPECTED_ARTIFACT_SIZE,
) -> dict[str, Any]:
    """Load, checksum, and validate the evaluated artifact once per process."""
    global _RUNTIME_ARTIFACT, _RUNTIME_PATH, _LOAD_COUNT
    artifact_path = Path(path) if path is not None else DEFAULT_ARTIFACT_PATH
    resolved = artifact_path.resolve() if artifact_path.exists() else artifact_path
    if (
        _RUNTIME_ARTIFACT is not None
        and _RUNTIME_PATH is not None
        and _RUNTIME_PATH == resolved
    ):
        return _RUNTIME_ARTIFACT
    verify_artifact_file(
        artifact_path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    try:
        loaded = joblib.load(artifact_path)
        payload = validate_artifact(loaded)
    except ArtifactIntegrityError:
        raise
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise DemoError(
            "The model artifact could not be loaded or is incompatible with "
            f"this application: {exc}"
        ) from exc
    _RUNTIME_ARTIFACT = payload
    _RUNTIME_PATH = resolved
    _LOAD_COUNT += 1
    return payload


def inject_runtime_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Install a validated in-memory artifact without touching the 66 MB file.

    Used by automated tests and an explicitly labelled synthetic UI fixture.
    """
    global _RUNTIME_ARTIFACT, _RUNTIME_PATH, _LOAD_COUNT
    payload = validate_artifact(artifact)
    _RUNTIME_ARTIFACT = payload
    _RUNTIME_PATH = None
    _LOAD_COUNT += 1
    return payload


def get_runtime_artifact() -> dict[str, Any]:
    if _RUNTIME_ARTIFACT is None:
        return load_evaluated_artifact()
    return _RUNTIME_ARTIFACT


def _coerce_image(image: Any) -> Image.Image:
    if image is None:
        raise DemoError("No image uploaded. Please upload a photo before classifying.")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise DemoError("No image uploaded. Please upload a photo before classifying.")
        if image.ndim not in {2, 3}:
            raise DemoError("The uploaded file is not a supported image array.")
        height, width = int(image.shape[0]), int(image.shape[1])
        _reject_unreasonable_size(width, height)
        try:
            return Image.fromarray(np.asarray(image).astype(np.uint8, copy=False))
        except (TypeError, ValueError) as exc:
            raise DemoError(
                "The uploaded file could not be interpreted as an image."
            ) from exc
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise DemoError(f"The image file was not found: {image_path}")
        try:
            with Image.open(image_path) as opened:
                opened.load()
                _reject_unreasonable_size(*opened.size)
                return opened.copy()
        except UnidentifiedImageError as exc:
            raise DemoError(
                "The uploaded file could not be read as an image. "
                "Use a JPEG, PNG, or other common still-image format."
            ) from exc
        except Image.DecompressionBombError as exc:
            raise DemoError(
                "The image is unreasonably large and was rejected before processing."
            ) from exc
        except OSError as exc:
            raise DemoError(
                "The uploaded file could not be read as an image. "
                "It may be corrupt or in an unsupported format."
            ) from exc
    if isinstance(image, Image.Image):
        try:
            image.load()
        except Exception as exc:
            raise DemoError(
                "The uploaded file could not be read as an image. "
                "It may be corrupt or in an unsupported format."
            ) from exc
        _reject_unreasonable_size(*image.size)
        return image
    raise DemoError("No image uploaded. Please upload a photo before classifying.")


def _reject_unreasonable_size(
    width: int,
    height: int,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
) -> None:
    if width <= 0 or height <= 0:
        raise DemoError("The uploaded image has invalid dimensions.")
    if width > max_side or height > max_side or (width * height) > max_pixels:
        raise DemoError(
            "The image is unreasonably large. Please upload a smaller photo "
            f"(each side at most {max_side} pixels, and at most {max_pixels} pixels total)."
        )


def format_prediction(result: PredictionResult) -> DemoView:
    rows = [
        [
            CLASS_DISPLAY_LABELS[class_name],
            f"{result.probabilities[class_name] * 100:.2f}%",
        ]
        for class_name in CANONICAL_CLASSES
    ]
    confidence = (
        "Calibrated model confidence: "
        f"{result.max_probability * 100:.1f}% "
        f"(frozen abstention threshold {FROZEN_ABSTENTION_THRESHOLD:.2f})"
    )
    display_class = CLASS_DISPLAY_LABELS[result.predicted_class]
    if result.abstained:
        return DemoView(
            primary_result=UNCERTAIN_PRIMARY,
            uncertainty_message=UNCERTAIN_PRIMARY,
            confidence_text=confidence,
            probability_rows=rows,
            tentative_class=f"Tentative top class (not accepted): {display_class}",
        )
    return DemoView(
        primary_result=display_class,
        uncertainty_message="",
        confidence_text=confidence,
        probability_rows=rows,
        tentative_class="",
    )


def classify_upload(
    image: Any,
    artifact: Any | None = None,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
) -> DemoView:
    """Classify an uploaded image through predict_image and format the result."""
    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        pil_image = _coerce_image(image)
        _reject_unreasonable_size(
            pil_image.size[0],
            pil_image.size[1],
            max_pixels=max_pixels,
            max_side=max_side,
        )
        loaded = get_runtime_artifact() if artifact is None else artifact
        try:
            result = predict_image(pil_image, loaded)
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                "The evaluated model file was not found. Place the preserved "
                f"artifact at {DEFAULT_ARTIFACT_PATH}."
            ) from exc
        except ValueError as exc:
            message = str(exc)
            lowered = message.lower()
            if "corrupt" in lowered or "unreadable" in lowered or "invalid" in lowered:
                raise DemoError(
                    "The uploaded file could not be read as an image. "
                    "It may be corrupt or in an unsupported format."
                ) from exc
            raise DemoError(
                "The model artifact is incompatible with this application: " + message
            ) from exc
        return format_prediction(result)
    except DemoError:
        raise
    except Image.DecompressionBombError as exc:
        raise DemoError(
            "The image is unreasonably large and was rejected before processing."
        ) from exc
    except UnidentifiedImageError as exc:
        raise DemoError(
            "The uploaded file could not be read as an image. "
            "Use a JPEG, PNG, or other common still-image format."
        ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = False
