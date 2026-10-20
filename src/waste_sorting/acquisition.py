"""Acquire the checksum-pinned, upstream TrashNet image archive."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

REVISION = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
ARCHIVE_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/garythung/trashnet/{REVISION}/data/dataset-resized.zip"
)
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
EXPECTED_COUNTS = dict(zip(CLASSES, (403, 501, 410, 594, 482, 137), strict=True))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract only recognized image members; never trust archive paths."""
    with zipfile.ZipFile(archive) as source:
        members = []
        names = set()
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if info.is_dir() or path.parts[0] == "__MACOSX" or path.name == ".DS_Store":
                continue
            if (
                len(path.parts) != 3
                or path.parts[0] != "dataset-resized"
                or path.parts[1] not in CLASSES
                or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
            ):
                raise ValueError(f"Unexpected archive member: {info.filename}")
            relative = Path(*path.parts[1:])
            if relative in names:
                raise ValueError(f"Repeated archive member: {relative}")
            names.add(relative)
            members.append((info, relative))
        if not members:
            raise ValueError("Archive contains no recognized images")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            actual = {p.relative_to(destination) for p in destination.rglob("*") if p.is_file()}
            if actual != names:
                raise ValueError("Existing dataset differs from archive; use a fresh destination")
            for info, relative in members:
                if sha256(destination / relative) != hashlib.sha256(source.read(info)).hexdigest():
                    raise ValueError(f"Existing source image was modified: {relative}")
            return
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "images"
            for info, relative in members:
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            staging.rename(destination)


def acquire(root: Path, archive: Path | None = None) -> dict:
    archive = archive or root / "data/raw/trashnet-dataset-resized.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".partial")
        try:
            with requests.get(SOURCE_URL, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if sha256(partial) != ARCHIVE_SHA256:
                raise ValueError("Downloaded archive checksum does not match pinned source")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive checksum does not match pinned source")
    extract_archive(archive, root / "data/raw/trashnet")
    report_dir = root / "docs/dataset"
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / "provenance.json"
    provenance = {
        "dataset": "TrashNet dataset-resized",
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": archive.stat().st_size,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "upstream_repository": "https://github.com/garythung/trashnet",
        "license_url": f"https://github.com/garythung/trashnet/blob/{REVISION}/LICENSE",
        "terms_note": "Upstream repository provides MIT terms; preserve author attribution.",
        "citation": (
            "Gary Thung and Mindy Yang (2016), Classification of Trash for Recyclability Status"
        ),
    }
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("archive_sha256") == ARCHIVE_SHA256:
            provenance["acquired_at_utc"] = previous["acquired_at_utc"]
    target.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
