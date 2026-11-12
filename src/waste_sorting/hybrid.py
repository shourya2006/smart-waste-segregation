"""Hybrid classical machine learning and deep learning models for waste sorting.

Combines pretrained CNN embeddings (MobileNetV2, EfficientNet-B0) with classical
classifiers (H1) and handcrafted colour/texture features (H2 early fusion).
Also produces Phase 1 SVM out-of-fold probabilities and H3 late probability fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from waste_sorting.dl_data import build_transforms, fold_image_ids
from waste_sorting.dl_training import load_trained_model
from waste_sorting.embeddings import load_embeddings
from waste_sorting.evaluation import (
    CLASSES,
    SELECTED_FAMILY,
    extract_split_features,
)
from waste_sorting.extended_features import extended_columns, extract_extended_features
from waste_sorting.models.svm import sqrt_features
from waste_sorting.training import _source_metadata


class BlockScaledFusionTransformer(BaseEstimator, TransformerMixin):
    """Transformer for early feature fusion.

    Applies StandardScaler to each feature block independently (fitted on fold-train only),
    then divides each block by sqrt(D_block) so each modality has equal total variance.
    """

    def __init__(self, dim_block1: int, dim_block2: int):
        self.dim_block1 = dim_block1
        self.dim_block2 = dim_block2
        self.scaler1_ = StandardScaler()
        self.scaler2_ = StandardScaler()

    def fit(self, X: np.ndarray, y: Any = None) -> BlockScaledFusionTransformer:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.dim_block1 + self.dim_block2:
            raise ValueError(
                f"Expected {self.dim_block1 + self.dim_block2} features, got {X.shape[1]}"
            )
        X1 = X[:, : self.dim_block1]
        X2 = X[:, self.dim_block1 :]
        self.scaler1_.fit(X1)
        self.scaler2_.fit(X2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X1 = X[:, : self.dim_block1]
        X2 = X[:, self.dim_block1 :]
        X1_scaled = self.scaler1_.transform(X1) / np.sqrt(self.dim_block1)
        X2_scaled = self.scaler2_.transform(X2) / np.sqrt(self.dim_block2)
        return np.hstack([X1_scaled, X2_scaled]).astype(np.float32)


def load_aligned_features(
    backbone: str,
    root: Path | str = ".",
    split: str = "train",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load pretrained CNN embeddings and Phase 1 extended handcrafted features.

    Asserts strict alignment of image IDs and order between the two representations.
    Returns:
        dict with:
            'embeddings': [N, 1280] float32
            'handcrafted': [N, 1608] float32 (selected columns)
            'labels': [N] int64
            'image_ids': [N] str
            'fused_raw': [N, 2888] float32
    """
    root = Path(root).resolve()
    if split != "train":
        raise ValueError(
            f"Hybrids can only be evaluated on 'train' across 5 CV folds; got {split!r}"
        )

    # 1. Load CNN embeddings
    emb_data = load_embeddings(backbone, split=split, root=root, config=config)
    emb_ids = emb_data["image_ids"]
    embeddings = emb_data["embeddings"]
    emb_labels = emb_data["labels"]

    # 2. Load Phase 1 extended features
    ext_cache_candidates = [
        root / "artifacts/phase2/extended-features/train.npz",
        root / "artifacts/phase1/extended-features/train.npz",
        root / "artifacts/daksh-rf-refinement/extended-training-features.npz",
    ]
    ext_cache = None
    for cand in ext_cache_candidates:
        if cand.is_file():
            ext_cache = cand
            break

    ext_matrix, ext_labels, ext_ids, _ = extract_split_features(
        split_name=split,
        root=root,
        cache_path=ext_cache,
    )

    # 3. Assert identity sets and strict ordering
    if len(emb_ids) != len(ext_ids):
        raise ValueError(
            f"Image ID count mismatch: embeddings={len(emb_ids)}, handcrafted={len(ext_ids)}"
        )
    if not np.array_equal(emb_ids, ext_ids):
        raise ValueError("Image IDs or ordering do not match between embeddings and handcrafted")
    if not np.array_equal(emb_labels, ext_labels):
        raise ValueError("Class labels do not match between embeddings and handcrafted features")

    # Select the Phase 1 winning 1,608 columns
    cols = extended_columns(SELECTED_FAMILY)
    handcrafted_selected = np.ascontiguousarray(ext_matrix[:, cols], dtype=np.float32)

    # Combined raw block [1608 + 1280 = 2888]
    fused_raw = np.hstack([handcrafted_selected, embeddings]).astype(np.float32)

    return {
        "embeddings": embeddings,
        "handcrafted": handcrafted_selected,
        "labels": emb_labels,
        "image_ids": emb_ids,
        "fused_raw": fused_raw,
        "embedding_metadata": emb_data.get("metadata", {}),
    }


def make_h1_pipeline(
    classifier_type: str,
    C: float = 1.0,
    gamma: Any = "scale",
    seed: int = 42,
) -> Pipeline:
    """Construct an H1 pipeline: StandardScaler -> Classifier on 1280-dim CNN embedding."""
    scaler = StandardScaler()
    if classifier_type == "logistic_regression":
        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    elif classifier_type == "linear_svm":
        clf = LinearSVC(
            C=C,
            dual="auto",
            max_iter=20000,
            random_state=seed,
        )
    elif classifier_type == "rbf_svm":
        clf = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
            cache_size=512,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type!r}")

    return Pipeline([("scaler", scaler), ("classifier", clf)])


def make_h2_pipeline(
    classifier_type: str,
    dim_hc: int = 1608,
    dim_cnn: int = 1280,
    C: float = 10.0,
    gamma: Any = "scale",
    seed: int = 42,
) -> Pipeline:
    """Construct an H2 pipeline: BlockScaledFusionTransformer -> Classifier on fused vector."""
    fusion_scaler = BlockScaledFusionTransformer(dim_block1=dim_hc, dim_block2=dim_cnn)
    if classifier_type == "rbf_svm":
        clf = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
            cache_size=512,
            random_state=seed,
        )
    elif classifier_type == "linear_svm":
        clf = LinearSVC(
            C=C,
            dual="auto",
            max_iter=20000,
            random_state=seed,
        )
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type!r}")

    return Pipeline([("scaler", fusion_scaler), ("classifier", clf)])


def run_cv(
    candidate_type: str,
    backbone: str,
    hyperparams: dict[str, Any],
    root: Path | str = ".",
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run 5-fold cross-validation across the frozen Phase 1 folds.

    Every transformation and estimator is fitted strictly on fold-train rows.
    Collects per-fold metrics, out-of-fold predictions, and fit timings.
    """
    if not (candidate_type.startswith("h1") or candidate_type.startswith("h2")):
        raise ValueError(f"Unknown candidate_type: {candidate_type!r}. Must be 'h1' or 'h2'.")

    root = Path(root).resolve()
    folds_path = root / "docs/training/folds.json"
    if not folds_path.is_file():
        raise FileNotFoundError(f"CV folds not found at {folds_path}")

    # Check output directory if provided
    out_path = Path(output_dir) if output_dir else None
    if out_path and out_path.exists():
        raise FileExistsError(f"Output directory already exists: {out_path}")

    # Load aligned representations
    features_bundle = load_aligned_features(backbone=backbone, root=root, split="train")
    all_ids = features_bundle["image_ids"]
    all_y = features_bundle["labels"]
    id_to_idx = {img_id: i for i, img_id in enumerate(all_ids)}

    if candidate_type.startswith("h1"):
        X_all = features_bundle["embeddings"]
    else:
        X_all = features_bundle["fused_raw"]

    # Per-fold storage
    fold_records: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    classes_list = list(CLASSES)
    n_classes = len(classes_list)

    total_fit_start = time.perf_counter()

    for fold in range(5):
        train_ids, val_ids = fold_image_ids(fold, folds_path)
        train_idx = np.array([id_to_idx[img_id] for img_id in train_ids])
        val_idx = np.array([id_to_idx[img_id] for img_id in val_ids])

        X_train, y_train = X_all[train_idx], all_y[train_idx]
        X_val, y_val = X_all[val_idx], all_y[val_idx]

        # Pipeline construction
        clf_type = hyperparams.get("classifier", "rbf_svm")
        C = float(hyperparams.get("C", 1.0))
        gamma = hyperparams.get("gamma", "scale")
        seed = int(hyperparams.get("seed", 42))

        if candidate_type.startswith("h1"):
            pipeline = make_h1_pipeline(clf_type, C=C, gamma=gamma, seed=seed)
        else:
            pipeline = make_h2_pipeline(
                clf_type,
                dim_hc=features_bundle["handcrafted"].shape[1],
                dim_cnn=features_bundle["embeddings"].shape[1],
                C=C,
                gamma=gamma,
                seed=seed,
            )

        # Fit strictly on fold training data
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # Predict on held-out fold
        y_pred = pipeline.predict(X_val)

        # Probability prediction if available
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_val)
        elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
            df = pipeline.decision_function(X_val)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            probs = np.zeros((len(val_idx), n_classes))
            for i, p in enumerate(y_pred):
                probs[i, p] = 1.0

        # Metrics
        f1_macro = float(f1_score(y_val, y_pred, average="macro"))
        acc = float(accuracy_score(y_val, y_pred))
        recalls = recall_score(y_val, y_pred, average=None, labels=range(n_classes))

        record: dict[str, Any] = {
            "fold": fold,
            "macro_f1": f1_macro,
            "accuracy": acc,
            "fit_seconds": fit_time,
        }
        for c_idx, c_name in enumerate(classes_list):
            record[f"recall_{c_name}"] = float(recalls[c_idx])
        fold_records.append(record)

        # OOF records
        for i, img_id in enumerate(val_ids):
            oof_item = {
                "image_id": img_id,
                "fold": fold,
                "label": classes_list[y_val[i]],
            }
            for c_idx, c_name in enumerate(classes_list):
                oof_item[f"prob_{c_name}"] = float(probs[i, c_idx])
            oof_rows.append(oof_item)

    total_time = time.perf_counter() - total_fit_start

    # Summary metrics across the 5 folds
    macro_f1s = [r["macro_f1"] for r in fold_records]
    accuracies = [r["accuracy"] for r in fold_records]

    mean_f1 = float(np.mean(macro_f1s))
    std_f1 = float(np.std(macro_f1s, ddof=1))  # Sample standard deviation
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies, ddof=1))

    mean_recalls = {
        c_name: float(np.mean([r[f"recall_{c_name}"] for r in fold_records]))
        for c_name in classes_list
    }

    summary: dict[str, Any] = {
        "candidate": candidate_type,
        "backbone": backbone,
        "hyperparameters": hyperparams,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_recalls": mean_recalls,
        "fold_macro_f1s": macro_f1s,
        "total_train_seconds": total_time,
        "embedding_metadata": features_bundle["embedding_metadata"],
        "provenance": _source_metadata(root),
    }

    runs_df = pd.DataFrame(fold_records)
    oof_df = pd.DataFrame(oof_rows).sort_values("image_id").reset_index(drop=True)

    if out_path:
        out_path.mkdir(parents=True, exist_ok=False)
        runs_df.to_csv(out_path / "runs.csv", index=False)
        oof_df.to_csv(out_path / "oof_predictions.csv", index=False)
        with (out_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return {
        "summary": summary,
        "runs": runs_df,
        "oof": oof_df,
    }


def generate_phase1_svm_oof(
    root: Path | str = ".",
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate out-of-fold calibrated probabilities for Phase 1 winning SVM.

    Reconstructs reconstruct_selected_pipeline() (RBF SVM on 1,608 extended features)
    and fits Platt sigmoid calibration strictly inside the training portion of each fold.
    Produces exactly 1,764 rows formatted as (image_id, fold, label, prob_<class>).
    """
    root = Path(root).resolve()
    folds_path = root / "docs/training/folds.json"
    if not folds_path.is_file():
        raise FileNotFoundError(f"CV folds not found at {folds_path}")

    # Check output path if provided
    out_file = Path(output_path) if output_path else None
    if out_file and out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    ext_cache_candidates = [
        root / "artifacts/phase2/extended-features/train.npz",
        root / "artifacts/phase1/extended-features/train.npz",
        root / "artifacts/daksh-rf-refinement/extended-training-features.npz",
    ]
    ext_cache = None
    for cand in ext_cache_candidates:
        if cand.is_file():
            ext_cache = cand
            break

    ext_matrix, ext_labels, ext_ids, _ = extract_split_features(
        split_name="train",
        root=root,
        cache_path=ext_cache,
    )

    cols = extended_columns(SELECTED_FAMILY)
    X_selected = np.ascontiguousarray(ext_matrix[:, cols], dtype=np.float32)

    id_to_idx = {img_id: i for i, img_id in enumerate(ext_ids)}
    classes_list = list(CLASSES)

    oof_rows: list[dict[str, Any]] = []

    def sqrt_transform(X):
        return np.sqrt(np.maximum(X, 0))

    for fold in range(5):
        train_ids, val_ids = fold_image_ids(fold, folds_path)
        train_idx = np.array([id_to_idx[img_id] for img_id in train_ids])
        val_idx = np.array([id_to_idx[img_id] for img_id in val_ids])

        X_train, y_train = X_selected[train_idx], ext_labels[train_idx]
        X_val, y_val = X_selected[val_idx], ext_labels[val_idx]

        pipe = Pipeline(
            [
                ("root", FunctionTransformer(sqrt_transform, validate=True)),
                ("normalize", Normalizer(norm="l2")),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma=10.0,
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)

        for i, img_id in enumerate(val_ids):
            item = {
                "image_id": img_id,
                "fold": fold,
                "label": classes_list[y_val[i]],
            }
            for c_idx, c_name in enumerate(classes_list):
                item[f"prob_{c_name}"] = float(probs[i, c_idx])
            oof_rows.append(item)

    oof_df = pd.DataFrame(oof_rows).sort_values("image_id").reset_index(drop=True)

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(out_file, index=False)

    return oof_df


def run_late_fusion(
    cnn_oof_path: Path | str,
    svm_oof_path: Path | str,
    output_dir: Path | str | None = None,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Execute H3 late probability fusion with nested cross-validation.

    Formulation: p_hybrid = w * p_cnn + (1 - w) * p_svm
    Nested selection: For each outer fold k in {0, 1, 2, 3, 4}:
      1. Choose w in {0.0, 0.1, ..., 1.0} maximizing Macro F1 on the other 4 folds' OOF rows.
      2. Evaluate outer fold k using its assigned w.
      3. Fold k never influences its own w.
    """
    cnn_path = Path(cnn_oof_path).resolve()
    svm_path = Path(svm_oof_path).resolve()

    if not cnn_path.is_file():
        raise FileNotFoundError(f"CNN OOF file not found: {cnn_path}")
    if not svm_path.is_file():
        raise FileNotFoundError(f"SVM OOF file not found: {svm_path}")

    out_dir = Path(output_dir).resolve() if output_dir else None
    if out_dir and out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    if weights is None:
        weights = [round(w * 0.1, 1) for w in range(11)]

    cnn_df = pd.read_csv(cnn_path)
    svm_df = pd.read_csv(svm_path)

    classes_list = list(CLASSES)
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    prob_cols = [f"prob_{c}" for c in classes_list]

    for col in ["image_id", "fold", "label"] + prob_cols:
        if col not in cnn_df.columns:
            raise ValueError(f"CNN OOF missing required column: {col}")
        if col not in svm_df.columns:
            raise ValueError(f"SVM OOF missing required column: {col}")

    # Standardize labels to integer class indices
    if pd.api.types.is_integer_dtype(cnn_df["label"]):
        cnn_labels = cnn_df["label"].to_numpy(dtype=int)
    else:
        cnn_labels = cnn_df["label"].map(class_to_idx).to_numpy(dtype=int)

    if pd.api.types.is_integer_dtype(svm_df["label"]):
        svm_labels = svm_df["label"].to_numpy(dtype=int)
    else:
        svm_labels = svm_df["label"].map(class_to_idx).to_numpy(dtype=int)

    cnn_clean = cnn_df.copy()
    cnn_clean["label_idx"] = cnn_labels
    svm_clean = svm_df.copy()
    svm_clean["label_idx"] = svm_labels

    # Verify identical image IDs
    cnn_ids = set(cnn_clean["image_id"])
    svm_ids = set(svm_clean["image_id"])
    if cnn_ids != svm_ids:
        raise ValueError(
            f"Image ID mismatch between CNN ({len(cnn_ids)}) and SVM ({len(svm_ids)}) OOF files"
        )

    merged = pd.merge(
        cnn_clean[["image_id", "fold", "label_idx"] + prob_cols],
        svm_clean[["image_id", "fold", "label_idx"] + prob_cols],
        on="image_id",
        suffixes=("_cnn", "_svm"),
    )

    if len(merged) != len(cnn_clean):
        raise ValueError("Merged length mismatch")

    if not (merged["fold_cnn"] == merged["fold_svm"]).all():
        raise ValueError("Fold assignment mismatch between CNN and SVM OOF files")

    if not (merged["label_idx_cnn"] == merged["label_idx_svm"]).all():
        raise ValueError("Label mismatch between CNN and SVM OOF files")

    folds = merged["fold_cnn"].to_numpy(dtype=int)
    y_true = merged["label_idx_cnn"].to_numpy(dtype=int)
    p_cnn = merged[[f"{c}_cnn" for c in prob_cols]].to_numpy(dtype=np.float32)
    p_svm = merged[[f"{c}_svm" for c in prob_cols]].to_numpy(dtype=np.float32)

    # Nested cross-validation
    chosen_w_per_fold: dict[int, float] = {}
    fold_records: list[dict[str, Any]] = []
    fused_probs = np.zeros_like(p_cnn)

    for fold in range(5):
        train_mask = folds != fold
        val_mask = folds == fold

        # Sweep w on training portion (other 4 folds)
        best_w = weights[0]
        best_train_f1 = -1.0
        for w in weights:
            p_train = w * p_cnn[train_mask] + (1.0 - w) * p_svm[train_mask]
            preds_train = np.argmax(p_train, axis=1)
            train_f1 = float(f1_score(y_true[train_mask], preds_train, average="macro"))
            if train_f1 > best_train_f1:
                best_train_f1 = train_f1
                best_w = w

        chosen_w_per_fold[fold] = best_w

        # Evaluate on held-out fold
        p_val = best_w * p_cnn[val_mask] + (1.0 - best_w) * p_svm[val_mask]
        fused_probs[val_mask] = p_val
        preds_val = np.argmax(p_val, axis=1)

        val_f1 = float(f1_score(y_true[val_mask], preds_val, average="macro"))
        val_acc = float(accuracy_score(y_true[val_mask], preds_val))
        val_recalls = recall_score(
            y_true[val_mask], preds_val, average=None, labels=range(len(classes_list))
        )

        record: dict[str, Any] = {
            "fold": fold,
            "chosen_w": best_w,
            "macro_f1": val_f1,
            "accuracy": val_acc,
        }
        for c_idx, c_name in enumerate(classes_list):
            record[f"recall_{c_name}"] = float(val_recalls[c_idx])
        fold_records.append(record)

    # Full dataset OOF predictions from fused probabilities' argmax
    all_preds = np.argmax(fused_probs, axis=1)
    cm = confusion_matrix(y_true, all_preds, labels=range(len(classes_list)))

    macro_f1s = [r["macro_f1"] for r in fold_records]
    accuracies = [r["accuracy"] for r in fold_records]
    mean_macro_f1 = float(np.mean(macro_f1s))
    std_macro_f1 = float(np.std(macro_f1s, ddof=1))
    mean_accuracy = float(np.mean(accuracies))

    mean_recalls: dict[str, float] = {}
    pooled_recalls: dict[str, dict[str, Any]] = {}
    for c_idx, c_name in enumerate(classes_list):
        mean_recalls[c_name] = float(np.mean([r[f"recall_{c_name}"] for r in fold_records]))
        c_mask = y_true == c_idx
        correct = int(np.sum(all_preds[c_mask] == c_idx))
        total = int(np.sum(c_mask))
        fmt = f"{correct} / {total} = {correct / total * 100:.2f}%" if total > 0 else "N/A"
        pooled_recalls[c_name] = {
            "correct": correct,
            "total": total,
            "fraction": float(correct / total) if total > 0 else 0.0,
            "formatted": fmt,
        }

    # Best global w across all 5 folds (for final model refitting)
    best_global_w = weights[0]
    best_global_f1 = -1.0
    for w in weights:
        p_all = w * p_cnn + (1.0 - w) * p_svm
        all_p = np.argmax(p_all, axis=1)
        score = float(f1_score(y_true, all_p, average="macro"))
        if score > best_global_f1:
            best_global_f1 = score
            best_global_w = w

    summary = {
        "candidate": "h3_late_fusion",
        "family": "h3",
        "chosen_w_per_fold": chosen_w_per_fold,
        "best_global_w": best_global_w,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_accuracy": mean_accuracy,
        "mean_recalls": mean_recalls,
        "pooled_recalls": pooled_recalls,
        "confusion_matrix": cm.tolist(),
        "classes": classes_list,
    }

    # Export OOF DataFrame
    oof_rows: list[dict[str, Any]] = []
    for i, row in merged.iterrows():
        item = {
            "image_id": row["image_id"],
            "fold": int(row["fold_cnn"]),
            "label": classes_list[row["label_idx_cnn"]],
        }
        for c_idx, c_name in enumerate(classes_list):
            item[f"prob_{c_name}"] = float(fused_probs[i, c_idx])
        oof_rows.append(item)
    oof_df = pd.DataFrame(oof_rows).sort_values("image_id").reset_index(drop=True)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(fold_records).to_csv(out_dir / "runs.csv", index=False)
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    return {
        "summary": summary,
        "runs": fold_records,
        "oof": oof_df,
    }


def fit_final_hybrid(
    name: str = "h3_late_fusion",
    root: Path | str = ".",
    output_dir: Path | str = "artifacts/phase2/hybrid-final/h3_late_fusion",
    a2_model_dir: Path | str = "artifacts/phase2/final/A2.1",
    w: float = 0.5,
) -> Path:
    """Fit final hybrid model on the full training split (1,764 images).

    Only fits the classical SVM component on the full training split; the deep
    CNN component reuses the final A2.1 model without retraining.
    Saves metadata.json recording model SHA-256 hashes, w parameter, and provenance.
    """
    root = Path(root).resolve()
    out_dir = Path(output_dir).resolve()
    a2_dir = Path(a2_model_dir).resolve()

    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    # Verify A2.1 model exists and is verified
    a2_model, a2_meta = load_trained_model(a2_dir)
    a2_hash = a2_meta["model_sha256"]

    # Reconstruct Phase 1 SVM on full training set
    ext_cache_candidates = [
        root / "artifacts/phase2/extended-features/train.npz",
        root / "artifacts/phase1/extended-features/train.npz",
        root / "artifacts/daksh-rf-refinement/extended-training-features.npz",
    ]
    ext_cache = None
    for cand in ext_cache_candidates:
        if cand.is_file():
            ext_cache = cand
            break

    ext_matrix, ext_labels, ext_ids, _ = extract_split_features(
        split_name="train",
        root=root,
        cache_path=ext_cache,
    )

    cols = extended_columns(SELECTED_FAMILY)
    X_selected = np.ascontiguousarray(ext_matrix[:, cols], dtype=np.float32)

    svm_pipeline = Pipeline(
        [
            ("root", FunctionTransformer(sqrt_features, validate=True)),
            ("normalize", Normalizer(norm="l2")),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma=10.0,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    svm_pipeline.fit(X_selected, ext_labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    svm_path = out_dir / "svm_pipeline.joblib"
    joblib.dump(svm_pipeline, svm_path)

    svm_hash = hashlib.sha256(svm_path.read_bytes()).hexdigest()

    metadata = {
        "name": name,
        "family": "h3",
        "w": float(w),
        "classes": list(CLASSES),
        "svm_model_sha256": svm_hash,
        "a2_model_dir": str(a2_dir.relative_to(root)),
        "a2_model_sha256": a2_hash,
        "selected_family": SELECTED_FAMILY,
        "num_training_samples": len(ext_labels),
        "provenance": _source_metadata(root),
    }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return out_dir


class HybridPredictor:
    """Wrapper encapsulating a fitted final hybrid model."""

    def __init__(
        self,
        metadata: dict[str, Any],
        cnn_model: torch.nn.Module,
        cnn_config: dict[str, Any],
        svm_pipeline: Pipeline,
    ):
        self.metadata = metadata
        self.cnn_model = cnn_model
        self.cnn_config = cnn_config
        self.svm_pipeline = svm_pipeline
        self.w = float(metadata["w"])
        self.classes = metadata["classes"]
        self.transform = build_transforms(cnn_config, train=False)


def load_hybrid(directory: Path | str, root: Path | str = ".") -> HybridPredictor:
    """Load a final hybrid model and verify all component SHA-256 hashes."""
    d = Path(directory).resolve()
    root = Path(root).resolve()
    meta_file = d / "metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Hybrid metadata not found: {meta_file}")

    meta = json.loads(meta_file.read_text())

    # Verify SVM hash
    svm_path = d / "svm_pipeline.joblib"
    if not svm_path.is_file():
        raise FileNotFoundError(f"SVM pipeline not found: {svm_path}")
    current_svm_hash = hashlib.sha256(svm_path.read_bytes()).hexdigest()
    if current_svm_hash != meta["svm_model_sha256"]:
        raise ValueError(
            f"SVM hash mismatch: expected {meta['svm_model_sha256']}, got {current_svm_hash}"
        )

    # Verify CNN model
    a2_path = root / meta["a2_model_dir"]
    cnn_model, a2_meta = load_trained_model(a2_path)
    if a2_meta["model_sha256"] != meta["a2_model_sha256"]:
        raise ValueError(
            f"A2.1 hash mismatch: expected {meta['a2_model_sha256']}, got {a2_meta['model_sha256']}"
        )

    svm_pipeline = joblib.load(svm_path)
    return HybridPredictor(
        metadata=meta,
        cnn_model=cnn_model,
        cnn_config=a2_meta["config"],
        svm_pipeline=svm_pipeline,
    )


def predict_proba(hybrid: HybridPredictor, images: list[Any]) -> np.ndarray:
    """Predict class probabilities [N, 6] for a list of PIL Images or image paths.

    Applies Om's eval transform to feed the CNN and Phase 1 feature extraction to feed the SVM,
    then combines them via late probability fusion: p = w * p_cnn + (1 - w) * p_svm.
    """
    if not images:
        return np.empty((0, len(hybrid.classes)), dtype=np.float32)

    pil_images: list[Image.Image] = []
    for item in images:
        if isinstance(item, Image.Image):
            pil_images.append(item.convert("RGB"))
        elif isinstance(item, (str, Path)):
            with Image.open(item) as img:
                pil_images.append(img.convert("RGB"))
        else:
            raise ValueError(f"Expected PIL Image or image path, got {type(item)}")

    # 1. CNN forward pass
    tensors = [hybrid.transform(img) for img in pil_images]
    batch_tensor = torch.stack(tensors)
    device = next(hybrid.cnn_model.parameters()).device
    batch_tensor = batch_tensor.to(device)

    hybrid.cnn_model.eval()
    with torch.no_grad():
        logits = hybrid.cnn_model(batch_tensor)
        p_cnn = torch.softmax(logits, dim=1).cpu().numpy()

    # 2. Phase 1 SVM predict_proba
    cols = extended_columns(hybrid.metadata["selected_family"])
    features_list = []
    for img in pil_images:
        ext_vec = extract_extended_features(img)
        sub_vec = ext_vec[cols]
        features_list.append(sub_vec)
    X_svm = np.stack(features_list).astype(np.float32)

    p_svm = hybrid.svm_pipeline.predict_proba(X_svm)

    # 3. Late fusion
    w = hybrid.w
    p_fused = w * p_cnn + (1.0 - w) * p_svm
    # Ensure exact row normalization
    row_sums = np.sum(p_fused, axis=1, keepdims=True)
    return (p_fused / row_sums).astype(np.float32)


def main() -> None:
    """CLI entry point for waste-hybrid."""
    parser = argparse.ArgumentParser(description="Hybrid classical ML + DL runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # waste-hybrid cv
    cv_parser = subparsers.add_parser("cv", help="Run 5-fold CV for a hybrid candidate.")
    cv_parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Candidate identifier (e.g. h1_mobilenet_v2_rbf, h2_efficientnet_b0_rbf, etc.)",
    )
    cv_parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["h1", "h2"],
        help="Explicit candidate family ('h1' or 'h2'). If omitted, inferred from name.",
    )
    cv_parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0"],
        help="CNN backbone embeddings to use",
    )
    cv_parser.add_argument(
        "--classifier",
        type=str,
        default="rbf_svm",
        choices=["rbf_svm", "linear_svm", "logistic_regression"],
        help="Classifier family",
    )
    cv_parser.add_argument("--C", type=float, default=1.0, help="Regularization C parameter")
    cv_parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="RBF gamma parameter (scale, 1e-3, 1e-4, etc.)",
    )
    cv_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path (refuses to overwrite existing)",
    )
    cv_parser.add_argument("--root", type=str, default=".", help="Repository root path")

    # waste-hybrid late-fusion
    fusion_parser = subparsers.add_parser(
        "late-fusion",
        help="Run H3 late probability fusion with nested CV across the 5 folds.",
    )
    fusion_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path (refuses to overwrite existing)",
    )
    fusion_parser.add_argument(
        "--cnn-oof",
        type=str,
        default="docs/phase-2/ablations/selected-oof_predictions.csv",
        help="CNN OOF predictions CSV path",
    )
    fusion_parser.add_argument(
        "--svm-oof",
        type=str,
        default="artifacts/phase2/hybrid/phase1_svm_oof.csv",
        help="Phase 1 SVM OOF predictions CSV path",
    )
    fusion_parser.add_argument("--root", type=str, default=".", help="Repository root path")

    # waste-hybrid phase1-oof
    oof_parser = subparsers.add_parser(
        "phase1-oof",
        help="Generate Phase 1 SVM out-of-fold probabilities across the 5 CV folds.",
    )
    oof_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path (refuses to overwrite existing)",
    )
    oof_parser.add_argument("--root", type=str, default=".", help="Repository root path")

    args = parser.parse_args()

    gamma_val: Any = args.gamma if hasattr(args, "gamma") else "scale"
    try:
        gamma_val = float(gamma_val)
    except ValueError:
        pass

    if args.command == "cv":
        if args.family is not None:
            cand_family = args.family
        else:
            warnings.warn(
                "Missing --family argument; inferring from candidate name prefix. "
                "Specify --family {h1,h2} explicitly in the future.",
                UserWarning,
                stacklevel=2,
            )
            cand_family = "h1" if "h1" in args.candidate else "h2"

        res = run_cv(
            candidate_type=cand_family,
            backbone=args.backbone,
            hyperparams={
                "candidate": args.candidate,
                "classifier": args.classifier,
                "C": args.C,
                "gamma": gamma_val,
                "seed": 42,
            },
            root=args.root,
            output_dir=args.output,
        )
        print(f"Candidate {args.candidate} completed successfully.")
        print(
            f"Mean Macro F1: {res['summary']['mean_macro_f1']:.4f} "
            f"+/- {res['summary']['std_macro_f1']:.4f}"
        )
        print(f"Mean Accuracy: {res['summary']['mean_accuracy']:.4f}")
        print("Per-class Recall:")
        for c, r in res["summary"]["mean_recalls"].items():
            print(f"  {c}: {r:.4f}")

    elif args.command == "late-fusion":
        root_path = Path(args.root).resolve()
        cnn_file = (
            root_path / args.cnn_oof if not Path(args.cnn_oof).is_absolute() else Path(args.cnn_oof)
        )
        svm_file = (
            root_path / args.svm_oof if not Path(args.svm_oof).is_absolute() else Path(args.svm_oof)
        )
        out_dir = (
            root_path / args.output if not Path(args.output).is_absolute() else Path(args.output)
        )
        res = run_late_fusion(cnn_oof_path=cnn_file, svm_oof_path=svm_file, output_dir=out_dir)
        summary = res["summary"]
        print("H3 Late Probability Fusion completed successfully.")
        print(f"Chosen w per fold: {summary['chosen_w_per_fold']}")
        print(f"Best Global w: {summary['best_global_w']}")
        print(
            f"Mean Macro F1: {summary['mean_macro_f1']:.4f} "
            f"+/- {summary['std_macro_f1']:.4f}"
        )
        print(f"Mean Accuracy: {summary['mean_accuracy']:.4f}")
        print("Mean per-fold Recall:")
        for c, r in summary["mean_recalls"].items():
            print(f"  {c}: {r:.4f}")

    elif args.command == "phase1-oof":
        df = generate_phase1_svm_oof(root=args.root, output_path=args.output)
        print(f"Phase 1 SVM OOF predictions successfully generated: shape={df.shape}")


if __name__ == "__main__":
    main()
