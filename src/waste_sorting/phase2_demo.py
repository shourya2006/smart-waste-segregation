"""Phase 2 model selector demo helpers.

Loads the three Phase 2 finalist models (H3 hybrid, A2.1 CNN, Phase 1 SVM),
verifies their SHA-256 hashes against the frozen protocol.json, applies
temperature scaling from protocol.json at inference time, and returns a
dict with calibrated probabilities, predicted class, max probability,
threshold, and abstention flag.

Loading is done once per process and reused; ``root`` is always passed
so the module works when installed non-editable in CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image

from waste_sorting.dl_data import build_transforms
from waste_sorting.dl_training import load_trained_model
from waste_sorting.extended_features import extended_columns, extract_extended_features
from waste_sorting.hybrid import load_hybrid
from waste_sorting.hybrid import predict_proba as hybrid_predict_proba
from waste_sorting.phase2_evaluation import apply_temperature_scaling

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_PATH = Path("docs/phase-2/evaluation/protocol.json")
A2_DIR = Path("artifacts/phase2/final/A2.1")
HYBRID_DIR = Path("artifacts/phase2/hybrid-final/h3_late_fusion")

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

# Display labels for model selector (must be stable — tests key on them)
SELECTOR_LABELS = {
    "h3_hybrid": "H3 hybrid (recommended)",
    "a2_1_cnn": "A2.1 CNN",
    "phase1_svm": "Phase 1 SVM",
}
LABEL_TO_KEY = {v: k for k, v in SELECTOR_LABELS.items()}
DEFAULT_MODEL_LABEL = SELECTOR_LABELS["h3_hybrid"]

# Process-level cache: keyed by model_name -> loaded model object (or pipeline)
_MODEL_CACHE: dict[str, Any] = {}
_PROTOCOL_CACHE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

class Phase2DemoError(Exception):
    """User-facing error for Phase 2 demo failures."""


class Phase2HashError(Phase2DemoError):
    """Raised when a model file is missing or its hash does not match protocol.json."""


def _load_protocol(root: Path) -> dict[str, Any]:
    """Load and cache protocol.json from *root*."""
    global _PROTOCOL_CACHE
    if _PROTOCOL_CACHE is not None:
        return _PROTOCOL_CACHE
    proto_path = root / PROTOCOL_PATH
    if not proto_path.is_file():
        raise Phase2DemoError(
            f"Frozen protocol file not found at {proto_path}. "
            "Ensure docs/phase-2/evaluation/protocol.json is present."
        )
    _PROTOCOL_CACHE = json.loads(proto_path.read_text(encoding="utf-8"))
    return _PROTOCOL_CACHE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hashes(model_key: str, protocol: dict[str, Any], root: Path) -> None:
    """Verify model files match hashes recorded in protocol.json.

    Raises Phase2HashError with a clear message on any mismatch.
    """
    expected_hashes: dict[str, Any] = (
        protocol.get("models", {}).get(model_key, {}).get("hashes", {})
    )
    if not expected_hashes:
        return  # Nothing to verify (should not happen with a frozen protocol)

    hybrid_dir = root / HYBRID_DIR
    a2_dir = root / A2_DIR

    svm_path = hybrid_dir / "svm_pipeline.joblib"
    hybrid_meta_path = hybrid_dir / "metadata.json"
    a2_pt_path = a2_dir / "model.pt"
    a2_meta_path = a2_dir / "metadata.json"

    # Compute lazily — only the ones that appear in expected_hashes
    def _hash(p: Path) -> str:
        if not p.is_file():
            raise Phase2HashError(
                f"Required model file not found: {p}. "
                "Ensure the Phase 2 artifacts are present."
            )
        return _sha256_file(p)

    for key, expected in expected_hashes.items():
        if key == "svm_pipeline_sha256":
            actual = _hash(svm_path)
        elif key == "hybrid_metadata_sha256":
            actual = _hash(hybrid_meta_path)
        elif key in ("model_pt_sha256", "recorded_a2_model_sha256"):
            actual = _hash(a2_pt_path)
        elif key == "a2_metadata_sha256":
            actual = _hash(a2_meta_path)
        elif key in ("selected_family", "w"):
            # Non-file entries — skip hash check
            continue
        else:
            # Unknown key type — skip silently to stay forward-compatible
            continue

        if actual != expected:
            raise Phase2HashError(
                f"Hash mismatch for {model_key}.{key}: "
                f"expected {expected!r}, got {actual!r}. "
                "The model file has changed; refusing to serve."
            )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_a2_cnn(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Load A2.1 CNN model (once) with hash verification."""
    a2_dir = root / A2_DIR
    _verify_hashes("a2_1_cnn", protocol, root)
    model, meta = load_trained_model(a2_dir)
    transform = build_transforms(meta["config"], train=False)
    return {
        "type": "a2_1_cnn",
        "model": model,
        "meta": meta,
        "transform": transform,
    }


def _load_h3_hybrid(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Load H3 hybrid model (once) with hash verification."""
    # load_hybrid does its own hash checks internally; we also check protocol hashes
    _verify_hashes("h3_hybrid", protocol, root)
    hybrid = load_hybrid(root / HYBRID_DIR, root=root)
    return {
        "type": "h3_hybrid",
        "hybrid": hybrid,
    }


def _load_phase1_svm(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Load Phase 1 SVM (from hybrid dir) with hash verification."""
    _verify_hashes("phase1_svm", protocol, root)
    hybrid_dir = root / HYBRID_DIR
    hybrid_meta = json.loads((hybrid_dir / "metadata.json").read_text(encoding="utf-8"))
    svm_path = hybrid_dir / "svm_pipeline.joblib"
    svm_pipeline = joblib.load(svm_path)
    return {
        "type": "phase1_svm",
        "svm_pipeline": svm_pipeline,
        "selected_family": hybrid_meta["selected_family"],
    }


def load_phase2_models(root: Path | str = ".") -> None:
    """Load all three Phase 2 models once per process and cache them.

    Call this at startup. After the first call, subsequent calls are no-ops.
    Raises Phase2DemoError / Phase2HashError if files are missing or hashes mismatch.
    """
    global _MODEL_CACHE
    root = Path(root).resolve()
    protocol = _load_protocol(root)

    if "h3_hybrid" not in _MODEL_CACHE:
        _MODEL_CACHE["h3_hybrid"] = _load_h3_hybrid(root, protocol)
    if "a2_1_cnn" not in _MODEL_CACHE:
        _MODEL_CACHE["a2_1_cnn"] = _load_a2_cnn(root, protocol)
    if "phase1_svm" not in _MODEL_CACHE:
        _MODEL_CACHE["phase1_svm"] = _load_phase1_svm(root, protocol)


def reset_phase2_models() -> None:
    """Clear the model cache. Intended for tests."""
    global _MODEL_CACHE, _PROTOCOL_CACHE
    _MODEL_CACHE = {}
    _PROTOCOL_CACHE = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _get_raw_probs(model_key: str, pil_image: Image.Image) -> np.ndarray:
    """Return raw (uncalibrated) [1, 6] probability array for *pil_image*."""
    entry = _MODEL_CACHE[model_key]

    if model_key == "h3_hybrid":
        hybrid = entry["hybrid"]
        probs = hybrid_predict_proba(hybrid, [pil_image])  # shape [1, 6]
        return probs.astype(np.float64)

    if model_key == "a2_1_cnn":
        transform = entry["transform"]
        model = entry["model"]
        tensor = transform(pil_image).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        p = p.astype(np.float64)
        row_sum = p.sum(axis=1, keepdims=True)
        return p / row_sum

    if model_key == "phase1_svm":
        svm_pipeline = entry["svm_pipeline"]
        selected_family = entry["selected_family"]
        cols = extended_columns(selected_family)
        rgb = pil_image.convert("RGB")
        ext_vec = extract_extended_features(rgb)
        X = np.stack([ext_vec[cols]]).astype(np.float32)
        probs = svm_pipeline.predict_proba(X).astype(np.float64)
        row_sum = probs.sum(axis=1, keepdims=True)
        return probs / row_sum

    raise Phase2DemoError(f"Unknown model key: {model_key!r}")


def predict_phase2(
    model_name: str,
    image: Any,
    root: Path | str = ".",
) -> dict[str, Any]:
    """Classify *image* using the named Phase 2 model.

    Parameters
    ----------
    model_name:
        One of the display labels (e.g. "H3 hybrid (recommended)") or
        the internal key (e.g. "h3_hybrid").
    image:
        A PIL Image, numpy array, or file path.
    root:
        Repository root (passed through to load_phase2_models if needed).

    Returns
    -------
    dict with keys:
        probabilities    : {class_name: float}  — calibrated, sum to 1
        predicted_class  : str
        max_probability  : float
        threshold        : float
        abstained        : bool
    """
    # Resolve display label -> internal key
    if model_name in LABEL_TO_KEY:
        model_key = LABEL_TO_KEY[model_name]
    elif model_name in SELECTOR_LABELS:
        model_key = model_name
    else:
        raise Phase2DemoError(
            f"Unknown model name {model_name!r}. "
            f"Expected one of {list(SELECTOR_LABELS.values())}."
        )

    root = Path(root).resolve()

    # Ensure models are loaded
    if model_key not in _MODEL_CACHE:
        load_phase2_models(root)

    protocol = _load_protocol(root)
    model_proto = protocol["models"][model_key]
    temperature: float = float(model_proto["temperature"])
    threshold: float = float(model_proto["demo_threshold"])

    # Coerce image to PIL
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8, copy=False))
    elif isinstance(image, (str, Path)):
        with Image.open(image) as img:
            pil_image = img.copy().convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise Phase2DemoError(f"Unsupported image type: {type(image)}")

    pil_image = pil_image.convert("RGB")

    # Raw probs -> temperature-scaled calibrated probs
    raw_probs = _get_raw_probs(model_key, pil_image)  # [1, 6]
    cal_probs = apply_temperature_scaling(raw_probs, temperature)  # [1, 6]
    cal_row = cal_probs[0]

    classes = tuple(protocol["classes"])
    prob_dict = {c: float(cal_row[i]) for i, c in enumerate(classes)}

    max_prob_idx = int(np.argmax(cal_row))
    predicted_class = classes[max_prob_idx]
    max_probability = float(cal_row[max_prob_idx])
    abstained = max_probability < threshold

    return {
        "probabilities": prob_dict,
        "predicted_class": predicted_class,
        "max_probability": max_probability,
        "threshold": threshold,
        "abstained": abstained,
    }
