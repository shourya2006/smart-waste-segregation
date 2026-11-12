"""Phase 2 calibration, validation abstention selection, and frozen protocol generation.

Calibrates the three Phase 2 finalists (Phase 1 SVM, A2.1 CNN, H3 hybrid) using
temperature scaling on training out-of-fold probabilities, selects each model's
abstention threshold on the validation split, and freezes all hashes, temperatures,
and thresholds in ``protocol.json`` before any test split evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from waste_sorting.acquisition import sha256
from waste_sorting.dl_data import (
    WasteImageDataset,
    build_transforms,
    load_config,
    load_manifest,
)
from waste_sorting.dl_training import load_trained_model
from waste_sorting.dl_training import predict as dl_predict
from waste_sorting.evaluation import (
    compute_calibration_curve_and_ece,
    compute_metrics,
    multiclass_brier_score,
    plot_reliability_diagram,
    select_abstention_threshold,
    sweep_abstention_thresholds,
)
from waste_sorting.extended_features import extended_columns, extract_extended_features
from waste_sorting.hybrid import load_hybrid
from waste_sorting.hybrid import predict_proba as hybrid_predict_proba
from waste_sorting.training import _source_metadata

MODEL_NAMES = ("phase1_svm", "a2_1_cnn", "h3_hybrid")
PROB_EPS = 1e-12
TEMP_BOUNDS = (0.05, 20.0)
TARGET_ACCURACY = 0.90
DEMO_TARGET_ACCURACY = 0.95
MIN_COVERAGE = 0.80


def _resolve_under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root / candidate)


def _collect_model_hashes(
    root: Path,
    a2_dir: Path,
    hybrid_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Collect and cross-verify SHA-256 hashes for all finalist model artifacts."""
    a2_meta_path = a2_dir / "metadata.json"
    a2_pt_path = a2_dir / "model.pt"
    if not a2_meta_path.is_file() or not a2_pt_path.is_file():
        raise FileNotFoundError(f"Missing A2.1 model files in {a2_dir}")

    a2_meta = json.loads(a2_meta_path.read_text(encoding="utf-8"))
    a2_pt_sha = sha256(a2_pt_path)
    if a2_pt_sha != a2_meta["model_sha256"]:
        raise ValueError(
            f"A2.1 model.pt SHA-256 mismatch: file={a2_pt_sha}, metadata={a2_meta['model_sha256']}"
        )

    hybrid_meta_path = hybrid_dir / "metadata.json"
    svm_path = hybrid_dir / "svm_pipeline.joblib"
    if not hybrid_meta_path.is_file() or not svm_path.is_file():
        raise FileNotFoundError(f"Missing H3 hybrid model files in {hybrid_dir}")

    hybrid_meta = json.loads(hybrid_meta_path.read_text(encoding="utf-8"))
    svm_sha = sha256(svm_path)
    if svm_sha != hybrid_meta["svm_model_sha256"]:
        raise ValueError(
            "Hybrid svm_pipeline.joblib SHA-256 mismatch: "
            f"file={svm_sha}, metadata={hybrid_meta['svm_model_sha256']}"
        )

    recorded_a2_dir = root / PureWindowsPath(hybrid_meta["a2_model_dir"]).as_posix()
    if recorded_a2_dir.resolve() != a2_dir.resolve():
        raise ValueError(
            f"Hybrid a2_model_dir ({recorded_a2_dir}) does not match {a2_dir.resolve()}"
        )
    if hybrid_meta["a2_model_sha256"] != a2_pt_sha:
        raise ValueError(
            "Hybrid recorded a2_model_sha256 mismatch: "
            f"hybrid={hybrid_meta['a2_model_sha256']}, a2_model_pt={a2_pt_sha}"
        )

    return {
        "phase1_svm": {
            "svm_pipeline_sha256": svm_sha,
            "hybrid_metadata_sha256": sha256(hybrid_meta_path),
            "selected_family": hybrid_meta["selected_family"],
        },
        "a2_1_cnn": {
            "model_pt_sha256": a2_pt_sha,
            "a2_metadata_sha256": sha256(a2_meta_path),
        },
        "h3_hybrid": {
            "svm_pipeline_sha256": svm_sha,
            "recorded_a2_model_sha256": hybrid_meta["a2_model_sha256"],
            "model_pt_sha256": a2_pt_sha,
            "hybrid_metadata_sha256": sha256(hybrid_meta_path),
            "w": float(hybrid_meta["w"]),
        },
    }


def _verify_protocol_hashes(
    model_name: str,
    protocol: dict[str, Any],
    root: Path,
    a2_dir: Path,
    hybrid_dir: Path,
) -> None:
    """Verify loaded artifacts match hashes recorded in protocol.json."""
    models_section = protocol.get("models")
    if not isinstance(models_section, dict) or model_name not in models_section:
        return
    expected_hashes = models_section[model_name].get("hashes")
    if not isinstance(expected_hashes, dict):
        return
    current_hashes = _collect_model_hashes(root, a2_dir, hybrid_dir)[model_name]
    for key, expected_val in expected_hashes.items():
        if key not in current_hashes or current_hashes[key] != expected_val:
            raise ValueError(
                f"Frozen hash mismatch for {model_name}.{key}: "
                f"expected {expected_val!r}, got {current_hashes.get(key)!r}"
            )


def load_and_validate_oof(
    oof_path: Path | str,
    root: Path | str,
    classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training OOF predictions and assert exact alignment with the 1,764 training IDs."""
    root = Path(root).resolve()
    oof_file = Path(oof_path)
    if not oof_file.is_file():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_file}")

    config = load_config(root / "configs/phase2.toml")
    manifest = load_manifest(root / config["paths"]["manifest"])
    train_df = manifest.loc[manifest["split"] == "train"].sort_values("image_id").reset_index(
        drop=True
    )

    oof_df = pd.read_csv(oof_file, keep_default_na=False)
    prob_cols = [f"prob_{c}" for c in classes]
    required_cols = {"image_id", "label", *prob_cols}
    missing_cols = required_cols - set(oof_df.columns)
    if missing_cols:
        raise ValueError(f"OOF file {oof_file} missing required columns: {sorted(missing_cols)}")

    if oof_df["image_id"].duplicated().any():
        raise ValueError(f"OOF file {oof_file} contains duplicate image_ids")

    expected_ids = set(train_df["image_id"])
    actual_ids = set(oof_df["image_id"])
    if actual_ids != expected_ids:
        extra = sorted(actual_ids - expected_ids)[:5]
        missing = sorted(expected_ids - actual_ids)[:5]
        raise ValueError(
            f"OOF file {oof_file} image_ids do not match training split "
            f"(count={len(oof_df)} vs expected={len(expected_ids)}, "
            f"extra_sample={extra}, missing_sample={missing})"
        )

    oof_sorted = oof_df.sort_values("image_id").reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    raw_labels = oof_sorted["label"]
    parsed_labels: list[int] = []
    for val in raw_labels:
        if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
            idx = int(val)
        elif isinstance(val, str) and val.isdigit():
            idx = int(val)
        elif str(val) in class_to_idx:
            idx = class_to_idx[str(val)]
        else:
            raise ValueError(f"Unknown label {val!r} in OOF file {oof_file}")
        if not 0 <= idx < len(classes):
            raise ValueError(f"Label index {idx} out of range in OOF file {oof_file}")
        parsed_labels.append(idx)

    y_oof = np.asarray(parsed_labels, dtype=np.int64)
    y_manifest = train_df["label"].map(class_to_idx).to_numpy(dtype=np.int64)
    if not np.array_equal(y_oof, y_manifest):
        raise ValueError(f"OOF labels in {oof_file} do not match manifest training labels")

    probs = oof_sorted[prob_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError(f"OOF file {oof_file} contains non-finite or negative probabilities")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"OOF probabilities in {oof_file} do not sum to 1.0")
    probs = probs / row_sums[:, None]

    return oof_sorted["image_id"].to_numpy(), y_oof, probs


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling ``softmax(log(clip(p, 1e-12)) / T)`` to probabilities."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Temperature must be positive and finite, got {temperature!r}")
    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape {probs_arr.shape}")
    clipped = np.clip(probs_arr, PROB_EPS, None)
    logits = np.log(clipped) / float(temperature)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def fit_temperature(
    y_true: np.ndarray,
    probs: np.ndarray,
    bounds: tuple[float, float] = TEMP_BOUNDS,
) -> float:
    """Fit scalar temperature T minimizing multiclass log loss via bounded optimization."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim != 2 or len(y_arr) != probs_arr.shape[0]:
        raise ValueError("Shape mismatch between y_true and probs")
    if len(y_arr) == 0:
        raise ValueError("Cannot fit temperature on empty sample")

    log_p = np.log(np.clip(probs_arr, PROB_EPS, None))
    row_indices = np.arange(len(y_arr))

    def objective(temp: float) -> float:
        scaled = log_p / temp
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(scaled), axis=1))
        correct_logits = scaled[row_indices, y_arr]
        return float(np.mean(log_sum_exp - correct_logits))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def evaluate_calibration_summary(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: tuple[str, ...],
) -> dict[str, float]:
    """Compute ECE, Brier score, and multiclass log loss for calibration comparison."""
    n_classes = len(classes)
    ece = compute_calibration_curve_and_ece(y_true, probs, n_bins=10)["ece"]
    brier = multiclass_brier_score(y_true, probs, n_classes=n_classes)
    nll = float(log_loss(y_true, probs, labels=list(range(n_classes))))
    return {
        "ece": float(ece),
        "brier_score": float(brier),
        "log_loss": float(nll),
    }


def predict_split(
    model_name: str,
    split: str,
    root: Path | str = ".",
    *,
    protocol_path: Path | str | None = None,
    a2_dir: Path | str | None = None,
    hybrid_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict ``(image_ids, labels, probs [N, 6])`` for ``validation`` or ``test``.

    Refuses ``split='test'`` unless a frozen ``protocol.json`` exists.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name {model_name!r}; expected one of {MODEL_NAMES}")
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")

    root = Path(root).resolve()
    resolved_a2 = (
        _resolve_under_root(root, a2_dir)
        if a2_dir is not None
        else root / "artifacts/phase2/final/A2.1"
    )
    resolved_hybrid = (
        _resolve_under_root(root, hybrid_dir)
        if hybrid_dir is not None
        else root / "artifacts/phase2/hybrid-final/h3_late_fusion"
    )

    if split == "test":
        proto_file = (
            _resolve_under_root(root, protocol_path)
            if protocol_path is not None
            else root / "docs/phase-2/evaluation/protocol.json"
        )
        if not proto_file.is_file():
            raise FileNotFoundError(
                f"Refusing split='test': frozen protocol file does not exist at {proto_file}"
            )
        protocol = json.loads(proto_file.read_text(encoding="utf-8"))
        _verify_protocol_hashes(model_name, protocol, root, resolved_a2, resolved_hybrid)

    config = load_config(root / "configs/phase2.toml")
    classes = tuple(config["classes"])
    manifest = load_manifest(root / config["paths"]["manifest"])
    split_df = manifest.loc[manifest["split"] == split].sort_values("image_id").reset_index(
        drop=True
    )
    if split_df.empty:
        raise ValueError(f"Split {split!r} has no included images")

    image_ids = split_df["image_id"].to_numpy()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = split_df["label"].map(class_to_idx).to_numpy(dtype=np.int64)
    image_root = root / config["paths"]["image_root"]
    image_paths = [image_root / img_id for img_id in image_ids]

    if model_name == "a2_1_cnn":
        model, a2_meta = load_trained_model(resolved_a2)
        if tuple(a2_meta["classes"]) != classes:
            raise ValueError("A2.1 model classes do not match configs/phase2.toml")
        eval_tf = build_transforms(a2_meta["config"], train=False)
        dataset = WasteImageDataset(manifest, image_root, classes, eval_tf, split=split)
        loader = DataLoader(
            dataset,
            batch_size=config["dl"]["data"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        pred_labels, probs, pred_ids, _ = dl_predict(model, loader, torch.device("cpu"))
        if tuple(pred_ids) != tuple(image_ids) or not np.array_equal(pred_labels, labels):
            raise ValueError("Dataset image ordering mismatch in A2.1 prediction")
        probs = probs.astype(np.float64)
        return image_ids, labels, probs / probs.sum(axis=1, keepdims=True)

    if model_name == "h3_hybrid":
        hybrid = load_hybrid(resolved_hybrid, root=root)
        if tuple(hybrid.classes) != classes:
            raise ValueError("H3 hybrid classes do not match configs/phase2.toml")
        batch_size = max(1, int(config["dl"]["data"]["batch_size"]))
        chunks = [
            hybrid_predict_proba(hybrid, image_paths[i : i + batch_size])
            for i in range(0, len(image_paths), batch_size)
        ]
        probs = np.vstack(chunks).astype(np.float64)
        return image_ids, labels, probs / probs.sum(axis=1, keepdims=True)

    # phase1_svm: SVM component of Daksh's fitted hybrid
    hybrid_meta = json.loads((resolved_hybrid / "metadata.json").read_text(encoding="utf-8"))
    svm_path = resolved_hybrid / "svm_pipeline.joblib"
    if sha256(svm_path) != hybrid_meta["svm_model_sha256"]:
        raise ValueError(f"SVM pipeline SHA-256 mismatch at {svm_path}")
    if tuple(hybrid_meta["classes"]) != classes:
        raise ValueError("Hybrid metadata classes do not match configs/phase2.toml")

    svm_pipeline = joblib.load(svm_path)
    cols = extended_columns(hybrid_meta["selected_family"])
    features_list = []
    for path in image_paths:
        with Image.open(path) as img:
            ext_vec = extract_extended_features(img.convert("RGB"))
        features_list.append(ext_vec[cols])
    X_svm = np.stack(features_list).astype(np.float32)
    probs = svm_pipeline.predict_proba(X_svm).astype(np.float64)
    return image_ids, labels, probs / probs.sum(axis=1, keepdims=True)


def calibrate_and_freeze(
    root: Path | str = ".",
    output_dir: Path | str = "docs/phase-2/evaluation",
    *,
    cnn_oof_path: Path | str = "docs/phase-2/ablations/selected-oof_predictions.csv",
    phase1_oof_path: Path | str = "artifacts/phase2/eval/phase1_svm_oof.csv",
    h3_oof_path: Path | str = "artifacts/phase2/eval/h3/oof_predictions.csv",
    a2_dir: Path | str = "artifacts/phase2/final/A2.1",
    hybrid_dir: Path | str = "artifacts/phase2/hybrid-final/h3_late_fusion",
) -> dict[str, Any]:
    """Fit temperatures on training OOFs, choose thresholds on validation, and freeze protocol."""
    root = Path(root).resolve()
    out_dir = _resolve_under_root(root, output_dir)
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(
            f"Frozen protocol already exists at {protocol_path}; refusing to overwrite."
        )

    config_path = root / "configs/phase2.toml"
    config = load_config(config_path)
    classes = tuple(config["classes"])
    resolved_a2 = _resolve_under_root(root, a2_dir)
    resolved_hybrid = _resolve_under_root(root, hybrid_dir)

    oof_paths = {
        "phase1_svm": _resolve_under_root(root, phase1_oof_path),
        "a2_1_cnn": _resolve_under_root(root, cnn_oof_path),
        "h3_hybrid": _resolve_under_root(root, h3_oof_path),
    }

    model_hashes = _collect_model_hashes(root, resolved_a2, resolved_hybrid)
    source_meta = _source_metadata(root)

    out_dir.mkdir(parents=True, exist_ok=True)
    models_protocol: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        oof_ids, y_oof, probs_oof = load_and_validate_oof(oof_paths[model_name], root, classes)
        oof_before = evaluate_calibration_summary(y_oof, probs_oof, classes)
        temperature = fit_temperature(y_oof, probs_oof, bounds=TEMP_BOUNDS)
        probs_oof_cal = apply_temperature_scaling(probs_oof, temperature)
        oof_after = evaluate_calibration_summary(y_oof, probs_oof_cal, classes)

        val_ids, y_val, val_probs_raw = predict_split(
            model_name,
            "validation",
            root,
            a2_dir=resolved_a2,
            hybrid_dir=resolved_hybrid,
        )
        val_uncal = evaluate_calibration_summary(y_val, val_probs_raw, classes)
        val_probs_cal = apply_temperature_scaling(val_probs_raw, temperature)

        sweep_df = sweep_abstention_thresholds(y_val, val_probs_cal)
        sweep_path = out_dir / f"validation-threshold-sweep-{model_name}.csv"
        sweep_df.to_csv(sweep_path, index=False)

        headline_selection = select_abstention_threshold(
            sweep_df,
            target_accuracy=TARGET_ACCURACY,
            min_coverage=MIN_COVERAGE,
        )
        headline_threshold = float(headline_selection["threshold"])
        demo_selection = select_abstention_threshold(
            sweep_df,
            target_accuracy=DEMO_TARGET_ACCURACY,
            min_coverage=MIN_COVERAGE,
        )
        demo_threshold = float(demo_selection["threshold"])

        val_metrics = compute_metrics(
            y_val, val_probs_cal, classes, threshold=headline_threshold
        )
        val_demo_metrics = compute_metrics(
            y_val, val_probs_cal, classes, threshold=demo_threshold
        )["abstention"]

        cal_curve = compute_calibration_curve_and_ece(y_val, val_probs_cal, n_bins=10)
        rel_path = out_dir / f"validation-reliability-{model_name}.png"
        plot_reliability_diagram(
            cal_curve,
            rel_path,
            title=f"Validation Reliability: {model_name} (T={temperature:.3f})",
        )

        oof_file = oof_paths[model_name]
        try:
            oof_rel = oof_file.resolve().relative_to(root).as_posix()
        except ValueError:
            oof_rel = str(oof_file)

        models_protocol[model_name] = {
            "hashes": model_hashes[model_name],
            "oof_path": oof_rel,
            "oof_sha256": sha256(oof_paths[model_name]),
            "oof_sample_count": int(len(oof_ids)),
            "temperature": temperature,
            "oof_calibration": {
                "before": oof_before,
                "after": oof_after,
            },
            "threshold": headline_threshold,
            "headline_threshold": headline_threshold,
            "demo_threshold": demo_threshold,
            "validation_sample_count": int(len(val_ids)),
            "validation_uncalibrated": val_uncal,
            "validation_threshold_selection": headline_selection,
            "headline_threshold_selection": headline_selection,
            "demo_threshold_selection": demo_selection,
            "validation_metrics": val_metrics,
            "validation_demo_abstention": val_demo_metrics,
            "validation_threshold_sweep_csv": sweep_path.name,
            "validation_reliability_png": rel_path.name,
        }

    protocol = {
        "scope": "Phase 2 Final Evaluation Protocol",
        "classes": list(classes),
        "calibration": {
            "method": "temperature_scaling_on_training_oof",
            "formula": "softmax(log(clip(p, 1e-12)) / T)",
            "optimizer": "scipy.optimize.minimize_scalar(method='bounded')",
            "temperature_bounds": list(TEMP_BOUNDS),
            "prob_clip_min": PROB_EPS,
        },
        "abstention": {
            "split": "validation",
            "target_accuracy": TARGET_ACCURACY,
            "demo_target_accuracy": DEMO_TARGET_ACCURACY,
            "min_coverage": MIN_COVERAGE,
            "headline_rule": {
                "target_accuracy": TARGET_ACCURACY,
                "min_coverage": MIN_COVERAGE,
            },
            "demo_rule": {
                "target_accuracy": DEMO_TARGET_ACCURACY,
                "min_coverage": MIN_COVERAGE,
            },
            "threshold_grid": "0.00..1.00 step 0.01",
            "note": (
                "demo_threshold added before the freeze because H3's headline threshold "
                "is 0.00 (no abstention); both are reported on test"
            ),
        },
        "limitations": [
            (
                "Temperatures were fitted on out-of-fold predictions of fold models, "
                "while the evaluated models are the final full-training-split models, "
                "as in Phase 1."
            ),
        ],
        "data_provenance": {
            "manifest_sha256": sha256(root / config["paths"]["manifest"]),
            "folds_sha256": sha256(root / config["paths"]["folds"]),
            "phase2_config_sha256": sha256(config_path),
        },
        "models": models_protocol,
        "planned_test_analyses": {
            "single_test_predictions_csv": "test-predictions.csv",
            "unconditional_metrics": [
                "macro_f1",
                "accuracy",
                "per_class_precision_recall_f1",
                "confusion_matrix",
                "ece",
                "brier_score",
                "log_loss",
            ],
            "selective_metrics": {
                "thresholds_reported": ["headline_threshold", "demo_threshold"],
                "metrics_per_threshold": [
                    "coverage",
                    "accepted_accuracy",
                    "accepted_macro_f1",
                ],
            },
            "bootstrap": {
                "metrics": ["macro_f1", "accuracy"],
                "confidence_level": 0.95,
                "n_resamples": 2000,
                "seed": 42,
            },
            "pairwise_mcnemar": {
                "pairs": [
                    ["h3_hybrid", "a2_1_cnn"],
                    ["h3_hybrid", "phase1_svm"],
                    ["a2_1_cnn", "phase1_svm"],
                ],
                "method": "exact_binomial_on_discordant_unconditional_predictions",
                "family_alpha": 0.05,
                "bonferroni_alpha": 0.05 / 3.0,
            },
            "phase1_reference_comparison": {
                "reference_doc": "docs/evaluation/README.md",
                "recorded_phase1_test_macro_f1": 0.8999,
            },
            "grad_cam": {
                "model": "a2_1_cnn",
                "target": "last_convolutional_block",
                "grid_count": 12,
                "selection": "6_correct_and_6_wrong_including_glass_metal_plastic_confusions",
            },
            "robustness_check": "evaluate_if_docs_robustness_exists_on_main",
        },
        "git_revision": source_meta["git_revision"],
        "source": source_meta,
    }

    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return protocol


def _print_calibration_summary(protocol: dict[str, Any]) -> None:
    print("Phase 2 Calibration & Validation Protocol Frozen")
    for model_name in MODEL_NAMES:
        entry = protocol["models"][model_name]
        before = entry["oof_calibration"]["before"]
        after = entry["oof_calibration"]["after"]
        val_u = entry["validation_metrics"]["unconditional"]
        head_sel = entry["headline_threshold_selection"]
        demo_sel = entry["demo_threshold_selection"]
        print(f"\n[{model_name}] T = {entry['temperature']:.6f}")
        print(
            f"  OOF before: ECE={before['ece']:.4f}, Brier={before['brier_score']:.4f}, "
            f"LogLoss={before['log_loss']:.4f}"
        )
        print(
            f"  OOF after : ECE={after['ece']:.4f}, Brier={after['brier_score']:.4f}, "
            f"LogLoss={after['log_loss']:.4f}"
        )
        print(
            f"  Val uncond: MacroF1={val_u['macro_f1']:.4f}, Acc={val_u['accuracy']:.4f}, "
            f"ECE={val_u['expected_calibration_error']:.4f}, "
            f"Brier={val_u['brier_score']:.4f}, LogLoss={val_u['log_loss']:.4f}"
        )
        recalls = " ".join(
            f"{c}={val_u['per_class'][c]['recall']:.4f}" for c in protocol["classes"]
        )
        print(f"  Val recall: {recalls}")
        print(
            f"  Headline  : {head_sel['threshold']:.2f} "
            f"(coverage={head_sel['coverage']:.4f} "
            f"[{head_sel['accepted_count']}/{head_sel['total_count']}], "
            f"accepted_acc={head_sel['accepted_accuracy']:.4f}, "
            f"target_met={head_sel['target_met']})"
        )
        print(
            f"  Demo (95%): {demo_sel['threshold']:.2f} "
            f"(coverage={demo_sel['coverage']:.4f} "
            f"[{demo_sel['accepted_count']}/{demo_sel['total_count']}], "
            f"accepted_acc={demo_sel['accepted_accuracy']:.4f}, "
            f"target_met={demo_sel['target_met']})"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 evaluation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Fit temperatures on training OOF, select validation thresholds, and freeze protocol.",
    )
    cal_parser.add_argument(
        "--output",
        default="docs/phase-2/evaluation",
        help="Output directory for protocol.json and validation artifacts",
    )
    cal_parser.add_argument("--root", default=".", help="Repository root")
    cal_parser.add_argument(
        "--cnn-oof",
        default="docs/phase-2/ablations/selected-oof_predictions.csv",
        help="A2.1 CNN training OOF CSV path",
    )
    cal_parser.add_argument(
        "--phase1-oof",
        default="artifacts/phase2/eval/phase1_svm_oof.csv",
        help="Phase 1 SVM training OOF CSV path",
    )
    cal_parser.add_argument(
        "--h3-oof",
        default="artifacts/phase2/eval/h3/oof_predictions.csv",
        help="H3 hybrid training OOF CSV path",
    )
    cal_parser.add_argument(
        "--a2-dir",
        default="artifacts/phase2/final/A2.1",
        help="Final A2.1 model directory",
    )
    cal_parser.add_argument(
        "--hybrid-dir",
        default="artifacts/phase2/hybrid-final/h3_late_fusion",
        help="Final H3 hybrid directory",
    )

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        protocol = calibrate_and_freeze(
            root=args.root,
            output_dir=args.output,
            cnn_oof_path=args.cnn_oof,
            phase1_oof_path=args.phase1_oof,
            h3_oof_path=args.h3_oof,
            a2_dir=args.a2_dir,
            hybrid_dir=args.hybrid_dir,
        )
        _print_calibration_summary(protocol)


if __name__ == "__main__":
    main()
