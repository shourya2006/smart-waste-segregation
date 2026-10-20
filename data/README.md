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

Record the source URL, retrieval date, source revision or dataset version,
archive checksum, dataset terms, and citation before the first experiment.
Credit Gary Thung and Mindy Yang and their 2016 report, *Classification of Trash
for Recyclability Status*.

The first data implementation should produce an audit and manifest containing
relative path, label, image dimensions, read status, file checksum, perceptual
hash, duplicate group, and split. Near-duplicate candidates need manual review.
Make duplicate decisions before creating stratified, group-aware splits.

Source images and generated data are ignored by Git. Keep the data acquisition
procedure, provenance summary, and final evaluation protocol in version control.
Split assignments and the approved dataset manifest should be preserved in a
small, deliberately tracked release location once their format is established.
