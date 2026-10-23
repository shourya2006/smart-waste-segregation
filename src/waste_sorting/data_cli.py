"""Command line entry points for the Phase 1 dataset milestone."""

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from waste_sorting.acquisition import acquire, sha256
from waste_sorting.dataset import (
    apply_reviews,
    assign_splits,
    audit_images,
    audit_summary,
    duplicate_candidates,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Acquire checksum-pinned TrashNet images")
    download.add_argument("--archive", type=Path, help="Use an already downloaded pinned archive")
    audit = sub.add_parser("audit", help="Audit images and propose duplicate review pairs")
    audit.add_argument("--phash-distance", type=int, default=6)
    sub.add_parser("split", help="Apply completed reviews and create fixed group-aware splits")
    sub.add_parser("eda", help="Create train-only exploratory charts and sample grids")
    feat = sub.add_parser(
        "features", help="Extract and cache handcrafted features from manifest"
    )
    feat.add_argument("--config", type=Path, default=None, help="Path to config TOML")
    feat.add_argument("--output", type=Path, default=None, help="Output .npz path")
    feat.add_argument(
        "--force", action="store_true", help="Force regeneration even if cache is up to date"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reports = root / "docs/dataset"
    reports.mkdir(parents=True, exist_ok=True)
    dataset = root / "data/raw/trashnet"
    if args.command == "download":
        print(json.dumps(acquire(root, args.archive), indent=2))
    elif args.command == "audit":
        frame = audit_images(dataset)
        candidates = duplicate_candidates(frame, args.phash_distance)
        frame.to_csv(reports / "audit_manifest.csv", index=False)
        candidates.to_csv(reports / "duplicate_candidates.csv", index=False)
        write_json(reports / "audit_summary.json", audit_summary(frame, candidates))
        write_json(
            reports / "audit_settings.json",
            {
                "phash_distance": args.phash_distance,
                "audit_manifest_sha256": sha256(reports / "audit_manifest.csv"),
                "candidate_sha256": sha256(reports / "duplicate_candidates.csv"),
            },
        )
        reviews_path = reports / "duplicate_reviews.csv"
        if not reviews_path.exists():
            reviews = candidates[["pair_id"]].copy()
            for field in ("decision", "reviewer", "reason"):
                reviews[field] = ""
            reviews.to_csv(reviews_path, index=False)
        from waste_sorting.eda import review_sheets

        sheets = review_sheets(root, candidates)
        print(f"Audited {len(frame)} files; {len(candidates)} pairs require review.")
        print(f"Created {len(sheets)} local review contact sheets.")
    elif args.command == "split":
        settings = json.loads((reports / "audit_settings.json").read_text())
        for filename, key in (
            ("audit_manifest.csv", "audit_manifest_sha256"),
            ("duplicate_candidates.csv", "candidate_sha256"),
        ):
            if sha256(reports / filename) != settings[key]:
                raise ValueError(f"Audit output changed; rerun audit: {filename}")
        frame = pd.read_csv(reports / "audit_manifest.csv", keep_default_na=False)
        candidates = pd.read_csv(reports / "duplicate_candidates.csv", keep_default_na=False)
        reviews = pd.read_csv(reports / "duplicate_reviews.csv", keep_default_na=False)
        observed_files = {
            p.relative_to(dataset).as_posix()
            for p in dataset.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        }
        if observed_files != set(frame.image_id):
            raise ValueError("Dataset file inventory changed since audit")
        for row in frame.itertuples():
            if sha256(dataset / row.image_id) != row.sha256:
                raise ValueError(f"Image changed since audit: {row.image_id}")
        config = tomllib.loads((root / "configs/phase1.toml").read_text())
        fractions = {key: config["splits"][key] for key in ("train", "validation", "test")}
        grouped = apply_reviews(frame, candidates, reviews)
        manifest = assign_splits(grouped, fractions, config["seed"])
        manifest_path = reports / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        valid = manifest[manifest.included]
        summary = {
            "seed": config["seed"],
            "target_fractions": fractions,
            "manifest_sha256": sha256(manifest_path),
            "review_sha256": sha256(reports / "duplicate_reviews.csv"),
            "counts": pd.crosstab(valid.label, valid.split).to_dict(),
            "totals": valid.split.value_counts().to_dict(),
            "groups": int(valid.duplicate_group.nunique()),
            "review_decisions": reviews.decision.value_counts().to_dict(),
            "leakage_checks": {
                field: bool(valid.groupby(field).split.nunique().max() == 1)
                for field in ("duplicate_group", "sha256", "pixel_sha256")
            },
            "limitation": "pHash review cannot guarantee all same-item views were detected",
        }
        write_json(reports / "split_summary.json", summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "eda":
        from waste_sorting.eda import create_eda

        create_eda(root)
    elif args.command == "features":
        from waste_sorting.features import build_feature_cache

        build_feature_cache(
            root=root,
            config_path=args.config,
            output_path=args.output,
            force=args.force,
        )


if __name__ == "__main__":
    main()
