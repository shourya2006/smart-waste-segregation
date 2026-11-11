"""Hybrid classical machine learning and deep learning models for waste sorting.

Combines pretrained CNN embeddings (MobileNetV2, EfficientNet-B0) with classical
classifiers (H1) and handcrafted colour/texture features (H2 early fusion).
Also produces Phase 1 SVM out-of-fold probabilities for Phase 2 PR B (H3 late fusion).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from waste_sorting.dl_data import fold_image_ids
from waste_sorting.embeddings import load_embeddings
from waste_sorting.evaluation import (
    CLASSES,
    SELECTED_FAMILY,
    extract_split_features,
)
from waste_sorting.extended_features import extended_columns
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
    elif candidate_type.startswith("h2"):
        X_all = features_bundle["fused_raw"]
    else:
        raise ValueError(f"Unknown candidate_type: {candidate_type!r}")

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
        cand_prefix = "h1" if "h1" in args.candidate else "h2"
        res = run_cv(
            candidate_type=cand_prefix,
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

    elif args.command == "phase1-oof":
        df = generate_phase1_svm_oof(root=args.root, output_path=args.output)
        print(f"Phase 1 SVM OOF predictions successfully generated: shape={df.shape}")


if __name__ == "__main__":
    main()
