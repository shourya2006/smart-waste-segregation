"""Runner and summary generator for the pre-registered Phase 2 DL ablation study.

Automates the 13 configurations defined in docs/phase-2/ablation-plan.md by driving
``waste-dl-train cv``. Supports resuming interrupted runs, subset execution (--only),
dry-run preview, and generating the consolidated summary table (summarise subcommand).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


# The 13 pre-registered configurations defined in docs/phase-2/ablation-plan.md
@dataclass(frozen=True)
class AblationConfig:
    id: str
    factor: str
    description: str
    overrides: tuple[str, ...]


ABLATIONS: tuple[AblationConfig, ...] = (
    AblationConfig(
        id="BASE",
        factor="Baseline",
        description="MobileNetV2, Frozen, 224, Drop 0.2, Aug Full, None",
        overrides=(),
    ),
    AblationConfig(
        id="A1.1",
        factor="Architecture",
        description="EfficientNet-B0, Frozen, 224, Drop 0.2, Aug Full, None",
        overrides=("dl.training.backbone=efficientnet_b0",),
    ),
    AblationConfig(
        id="A2.1",
        factor="Fine-Tuning",
        description="MobileNetV2, Last 2 Blocks, 224, Drop 0.2, Aug Full, None",
        overrides=("dl.training.unfreeze_blocks=2",),
    ),
    AblationConfig(
        id="A3.1",
        factor="Resolution",
        description="MobileNetV2, Frozen, 160, Drop 0.2, Aug Full, None",
        overrides=("preprocessing.width=160", "preprocessing.height=160"),
    ),
    AblationConfig(
        id="A4.0",
        factor="Dropout",
        description="MobileNetV2, Frozen, 224, Drop 0.0, Aug Full, None",
        overrides=("dl.training.dropout=0.0",),
    ),
    AblationConfig(
        id="A4.2",
        factor="Dropout",
        description="MobileNetV2, Frozen, 224, Drop 0.5, Aug Full, None",
        overrides=("dl.training.dropout=0.5",),
    ),
    AblationConfig(
        id="A5.1",
        factor="Augmentation",
        description="No Augmentation (All Off)",
        overrides=("dl.augmentation.enabled=false",),
    ),
    AblationConfig(
        id="A5.2",
        factor="Augmentation",
        description="No Random Resized Crop",
        overrides=("dl.augmentation.random_resized_crop=false",),
    ),
    AblationConfig(
        id="A5.3",
        factor="Augmentation",
        description="No Horizontal Flip",
        overrides=("dl.augmentation.horizontal_flip=false",),
    ),
    AblationConfig(
        id="A5.4",
        factor="Augmentation",
        description="No Random Rotation",
        overrides=("dl.augmentation.rotation=false",),
    ),
    AblationConfig(
        id="A5.5",
        factor="Augmentation",
        description="No Color Jitter",
        overrides=("dl.augmentation.color_jitter=false",),
    ),
    AblationConfig(
        id="A6.1",
        factor="Imbalance",
        description="Class-Weighted Cross-Entropy",
        overrides=("dl.training.class_weighted_loss=true",),
    ),
    AblationConfig(
        id="A6.2",
        factor="Imbalance",
        description="Balanced Random Sampling",
        overrides=("dl.data.balanced_sampling=true",),
    ),
)


def get_command_for_ablation(
    ablation: AblationConfig,
    runs_dir: Path,
    use_cli: bool = True,
    python_exe: str = sys.executable,
) -> list[str]:
    """Build the command line arguments to execute one ablation run."""
    output_dir = runs_dir / ablation.id
    if use_cli:
        cmd = ["waste-dl-train", "cv", "--output", str(output_dir)]
    else:
        cmd = [python_exe, "-m", "waste_sorting.dl_training", "cv", "--output", str(output_dir)]
    for override in ablation.overrides:
        cmd.extend(["--set", override])
    return cmd


def handle_run_directory(output_dir: Path, *, dry_run: bool = False) -> bool:
    """Inspect output directory status.

    Returns:
        True if the run is already complete and should be skipped.
        False if the run needs to be executed (or re-executed after renaming).
    """
    if not output_dir.exists():
        return False

    summary_file = output_dir / "summary.json"
    if summary_file.is_file():
        # Completed run
        return True

    # Incomplete / interrupted run
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    interrupted_dir = output_dir.with_name(f"{output_dir.name}.interrupted-{timestamp}")
    print(f"Found interrupted run in {output_dir}; renaming to {interrupted_dir}")
    if not dry_run:
        output_dir.rename(interrupted_dir)
    return False


def run_ablations(
    runs_dir: Path,
    only_ids: list[str] | None = None,
    dry_run: bool = False,
    python_exe: str = sys.executable,
) -> None:
    """Run all or selected ablation configurations."""
    runs_dir = Path(runs_dir)
    if not dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(only_ids) if only_ids else None
    if selected_ids:
        valid_ids = {a.id for a in ABLATIONS}
        invalid = selected_ids - valid_ids
        if invalid:
            raise ValueError(
                f"Unknown ablation ID(s): {sorted(invalid)}. Valid: {sorted(valid_ids)}"
            )

    configs_to_process = [
        a for a in ABLATIONS if selected_ids is None or a.id in selected_ids
    ]

    for ablation in configs_to_process:
        output_dir = runs_dir / ablation.id
        is_done = handle_run_directory(output_dir, dry_run=dry_run)
        if is_done:
            print(f"[{ablation.id}] Completed summary.json exists; skipping.")
            continue

        cmd = get_command_for_ablation(ablation, runs_dir, python_exe=python_exe)
        print(f"[{ablation.id}] {'(dry-run) ' if dry_run else ''}Running: {' '.join(cmd)}")
        if dry_run:
            continue

        # Execute training subprocess and stream output
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Ablation run {ablation.id} failed with exit code {proc.returncode}"
            )


def summarise_ablations(
    runs_dir: Path,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Read all completed runs and generate the summary table adhering to the plan."""
    runs_dir = Path(runs_dir)
    rows: list[dict[str, object]] = []

    for ablation in ABLATIONS:
        run_folder = runs_dir / ablation.id
        summary_file = run_folder / "summary.json"
        runs_file = run_folder / "runs.csv"

        if not summary_file.is_file() or not runs_file.is_file():
            print(f"Warning: {ablation.id} is missing summary.json or runs.csv in {runs_dir}")
            continue

        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)

        runs_df = pd.read_csv(runs_file)

        # Compute per-fold metrics across completed folds
        classes = summary.get("config", {}).get(
            "classes", ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        )
        recall_cols = [f"recall_{c}" for c in classes]

        if "median_best_epoch" not in summary:
            raise KeyError(
                f"Run '{ablation.id}' summary file {summary_file} is missing 'median_best_epoch'"
            )
        median_best = int(summary["median_best_epoch"])

        row: dict[str, object] = {
            "id": ablation.id,
            "factor": ablation.factor,
            "description": ablation.description,
            "mean_macro_f1": float(summary["mean_macro_f1"]),
            "std_macro_f1": float(summary["std_macro_f1"]),
            "mean_accuracy": float(summary["mean_accuracy"]),
            "median_best_epoch": median_best,
        }
        for col in recall_cols:
            val = float(runs_df[col].mean()) if col in runs_df.columns else float("nan")
            row[f"mean_{col}"] = val

        mean_epochs = (
            float(runs_df["epochs_run"].mean()) if "epochs_run" in runs_df else float("nan")
        )
        mean_seconds = (
            float(runs_df["train_seconds"].mean()) if "train_seconds" in runs_df else float("nan")
        )
        row["mean_epochs_run"] = mean_epochs
        row["mean_train_seconds"] = mean_seconds
        row["device"] = str(summary.get("device", "unknown"))
        row["trainable_parameters"] = int(summary.get("trainable_parameters", 0))

        rows.append(row)

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Wrote summary ({len(df)} rows) to {output_csv}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 DL ablation runner and summariser.")
    subparsers = parser.add_subparsers(dest="subcommand")

    # Runner sub-command (or default if no subcommand given)
    run_parser = subparsers.add_parser("run", help="Run ablation experiments")
    run_parser.add_argument(
        "--runs-dir",
        default="artifacts/phase2/runs",
        help="Directory to save individual ablation runs (default: artifacts/phase2/runs)",
    )
    run_parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="Run only specified ablation IDs (e.g. --only BASE A1.1)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    # Summarise sub-command
    summary_parser = subparsers.add_parser(
        "summarise", help="Generate summary table from completed runs"
    )
    summary_parser.add_argument(
        "--runs-dir",
        default="artifacts/phase2/runs",
        help="Directory containing individual ablation run folders",
    )
    summary_parser.add_argument(
        "--output-csv",
        default="docs/phase-2/ablations/summary.csv",
        help="Destination path for summary CSV (default: docs/phase-2/ablations/summary.csv)",
    )

    args, unknown = parser.parse_known_args()

    # If no subcommand passed, default to running (supporting top-level flags like --dry-run)
    if args.subcommand is None:
        default_parser = argparse.ArgumentParser(description="Phase 2 DL ablation runner.")
        default_parser.add_argument(
            "--runs-dir",
            default="artifacts/phase2/runs",
            help="Directory to save individual ablation runs",
        )
        default_parser.add_argument(
            "--only",
            nargs="+",
            metavar="ID",
            help="Run only specified ablation IDs",
        )
        default_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print commands without executing them",
        )
        run_args = default_parser.parse_args(unknown)
        run_ablations(
            runs_dir=Path(run_args.runs_dir),
            only_ids=run_args.only,
            dry_run=run_args.dry_run,
        )
    elif args.subcommand == "run":
        run_ablations(
            runs_dir=Path(args.runs_dir),
            only_ids=args.only,
            dry_run=args.dry_run,
        )
    elif args.subcommand == "summarise":
        summarise_ablations(
            runs_dir=Path(args.runs_dir),
            output_csv=Path(args.output_csv),
        )


if __name__ == "__main__":
    main()
