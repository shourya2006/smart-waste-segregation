# Dataset preparation

Use [TrashNet](https://github.com/garythung/trashnet), following the download link
provided by its official README. Do not reuse its historical Lua training code;
this project will implement the Phase 1 pipeline in Python.

Place the original class folders under `data/raw/trashnet/`:

```text
data/raw/trashnet/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

The official README lists 2,527 images: paper 594, glass 501, plastic 482,
metal 410, cardboard 403, and trash 137. Verify the downloaded copy and record
any differences. Preserve the source images unchanged.

The downloader records source URL, retrieval date, source revision, archive
checksum, dataset terms, and citation in `docs/dataset/provenance.json`.
Credit Gary Thung and Mindy Yang and their 2016 report, *Classification of Trash
for Recyclability Status*.

The data commands produce an audit and manifest containing
relative path, label, image dimensions, read status, file checksum, perceptual
hash, duplicate group, and split. The 45 near-duplicate candidates have a recorded
visual review in `docs/dataset/duplicate_reviews.csv`; six uncertain pairs are
grouped conservatively. Review decisions were completed before split creation.

Three identical-content pairs have conflicting class labels. All six images are
excluded from modeling while their source files remain unchanged. See
`docs/dataset/audit_manifest.csv` for the exact rows and
`docs/dataset/audit_summary.json` for totals.

Source images and generated data are ignored by Git. Keep the data acquisition
procedure, provenance summary, and final evaluation protocol in version control.
The approved manifest, split assignments, duplicate review decisions, provenance,
audit summaries, and train-only EDA are deliberately tracked under `docs/dataset/`.
