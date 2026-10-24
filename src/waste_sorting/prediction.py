"""Prediction interface for calibrated waste classification with abstention.

Uses the shared extended feature extractor and saved calibrated ensemble.
Validates artifact metadata, feature dimensions, class order and input images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from waste_sorting.extended_features import (
    DIMENSION as EXTENDED_DIMENSION,
)
from waste_sorting.extended_features import (
    extract_extended_features,
)


@dataclass(frozen=True)
class PredictionResult:
    """Structured result returned by predict_image."""

    predicted_class: str
    probabilities: dict[str, float]
    max_probability: float
    abstained: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "probabilities": dict(self.probabilities),
            "max_probability": self.max_probability,
            "abstained": self.abstained,
        }

    def __getitem__(self, item: str) -> Any:
        return self.as_dict()[item]

    def __contains__(self, item: str) -> bool:
        return item in self.as_dict()

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()


def validate_artifact(artifact: Any) -> dict[str, Any]:
    """Validate artifact structure, metadata, classes, dimensions and threshold."""
    if isinstance(artifact, (str, Path)):
        artifact_path = Path(artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        loaded = joblib.load(artifact_path)
    elif isinstance(artifact, dict):
        loaded = artifact
    else:
        raise ValueError(
            "Expected artifact to be a dict or a filesystem path to a joblib file, "
            f"got {type(artifact)}"
        )

    required_keys = {
        "calibrated_model",
        "classes",
        "threshold",
        "input_dimension",
    }
    missing = required_keys - set(loaded.keys())
    if missing:
        raise ValueError(f"Artifact missing required metadata keys: {sorted(missing)}")

    classes = loaded["classes"]
    if not isinstance(classes, (list, tuple)) or len(classes) != 6 or len(set(classes)) != 6:
        raise ValueError(f"Artifact must define exactly 6 distinct classes, got {classes}")

    input_dim = loaded["input_dimension"]
    if input_dim != EXTENDED_DIMENSION:
        raise ValueError(
            f"Artifact input dimension mismatch: expected {EXTENDED_DIMENSION}, got {input_dim}. "
            "The 1,806-feature baseline cache cannot be substituted for the extended feature model."
        )

    threshold = loaded["threshold"]
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Artifact threshold must be a float in [0.0, 1.0], got {threshold}")

    model = loaded["calibrated_model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Calibrated model inside artifact does not provide predict_proba")

    return loaded


def predict_image(
    image: Image.Image | str | Path,
    artifact: Any,
) -> PredictionResult:
    """Predict waste class with calibrated probabilities and abstention.

    Args:
        image: PIL Image object, or path to an image file.
        artifact: Loaded artifact dictionary or path to saved artifact .joblib file.

    Returns:
        PredictionResult with predicted_class, probabilities (mapped to saved class order),
        max_probability, and abstained flag.
    """
    valid_artifact = validate_artifact(artifact)
    model = valid_artifact["calibrated_model"]
    classes = list(valid_artifact["classes"])
    threshold = float(valid_artifact["threshold"])

    # Load and validate PIL Image
    pil_image: Image.Image
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as opened:
                pil_image = opened.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Corrupt or unreadable image file {image_path}: {exc}") from exc
    elif isinstance(image, Image.Image):
        try:
            # Force verify / load image data
            image.load()
            pil_image = image
        except Exception as exc:
            raise ValueError(f"Invalid or corrupt PIL Image: {exc}") from exc
    else:
        raise ValueError(f"Expected PIL Image or image path, got {type(image)}")

    # Extract extended features (2,760 dimensions)
    try:
        feature_vector = extract_extended_features(pil_image)
    except Exception as exc:
        raise ValueError(f"Feature extraction failed on input image: {exc}") from exc

    if feature_vector.shape != (EXTENDED_DIMENSION,):
        raise ValueError(
            f"Extracted feature dimension {feature_vector.shape} != ({EXTENDED_DIMENSION},). "
            "Do not substitute the original 1,806-feature extractor."
        )
    if not np.isfinite(feature_vector).all():
        raise ValueError("Extracted feature vector contains non-finite values.")

    # Predict calibrated class probabilities
    probs = model.predict_proba(feature_vector.reshape(1, -1))[0]
    if probs.shape != (len(classes),):
        raise ValueError(
            f"Predicted probability shape {probs.shape} does not match class count {len(classes)}"
        )
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError("Predicted probabilities contain non-finite or negative values")
    if not np.isclose(np.sum(probs), 1.0, atol=1e-5):
        raise ValueError(f"Predicted probabilities do not sum to 1.0: sum={np.sum(probs)}")

    # Map probabilities to saved class order
    probability_map = {cls_name: float(probs[idx]) for idx, cls_name in enumerate(classes)}
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[max_idx])
    predicted_class = classes[max_idx]
    abstained = bool(max_prob < threshold)

    return PredictionResult(
        predicted_class=predicted_class,
        probabilities=probability_map,
        max_probability=max_prob,
        abstained=abstained,
    )
