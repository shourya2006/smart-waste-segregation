# External robustness set: everyday waste photos

TrashNet photos were all taken against a white posterboard. That was the biggest limitation in
the Phase 2 report. This set measures how much accuracy holds up on ordinary photos: real
backgrounds, real lighting, several phones. It is an **external** test set. It is scored
**once** with the models frozen in
[`protocol.json`](../phase-2/evaluation/protocol.json). Nothing is ever tuned, recalibrated,
reselected or trained on it. Results: [docs/phase-2/robustness](../phase-2/robustness/README.md).

Collected by Om Yadav, with photos from 2–3 teammates' phones.

## How the photos were taken

- **Classes:** the six TrashNet classes from `configs/phase2.toml`: cardboard, glass, metal,
  paper, plastic, trash. The target is at least 50 photos per class and at least 300 in total.
- **One item per photo**, made of one clearly visible material. Mixed-material, dirty or
  ambiguous items were skipped, not guessed.
- **Real conditions**, recorded for every photo in [`conditions.csv`](conditions.csv):
  - `background`: `table`, `floor`, `cloth` or `grass`
  - `lighting`: `daylight`, `indoor` or `dim`
  - `device`: `phone_a`, `phone_b` or `phone_c` (a code for the phone, not a name)
- **No bursts.** Each item was photographed once or twice, never in a burst. The audit flags
  near-identical photos.
- Photos are stored as `data/raw/robustness/<class>/<file>`. Phone HEIC files must be exported
  as JPEG first, because Pillow cannot open HEIC and the audit would mark them unreadable.

## Privacy rules

- No people, faces, hands with rings or tattoos, screens, letters, labels with names or
  addresses, or other personal information in the frame. Every photo was checked by eye.
- No location data. Take the photos with the camera's location setting off. The audit reads
  each file's EXIF and **excludes** any photo with a GPS block. Those photos are retaken, not
  edited.
- Devices are recorded only as `phone_a`/`phone_b`/`phone_c`.
- **The images are never committed to Git** (`data/raw/*` is ignored). They are shared only as
  a zip on the team Drive.

## Drive archive

| Archive | Contents | SHA-256 |
| --- | --- | --- |
| `robustness-set-v1.zip` | `robustness/<class>/<photo>` | `TODO: fill in after zipping` |

Create and hash the archive from the repository root:

```sh
cd data/raw && zip -r -X ../../robustness-set-v1.zip robustness -x '*.DS_Store' && cd ../..
shasum -a 256 robustness-set-v1.zip
```

To restore it, unzip so the photos land at `data/raw/robustness/<class>/`, then run the audit.
The manifest's per-photo SHA-256 values confirm that the photos match.

## Audit

```sh
waste-data robustness-audit
```

The audit does these things:

- opens every photo (EXIF orientation applied) and records its label, width, height, SHA-256
  and perceptual hash;
- warns about GPS EXIF data and excludes those photos (`gps_exif`);
- flags near-duplicates **within the set** (`duplicate_content`, `near_duplicate_within_set`;
  the first photo by file name is kept) and **against all 2,527 TrashNet images**
  (`exact_duplicate_trashnet`, `near_duplicate_trashnet`). Both checks use the Phase 1 pHash
  distance of 6 from `docs/dataset/audit_settings.json`;
- refuses to run if the photos are inside `data/raw/trashnet`, so the training loaders can
  never pick them up;
- writes [`manifest.csv`](manifest.csv) with the columns `image_id, label, sha256, phash, width,
  height, background, lighting, device, included, exclusion_reason`. It also writes
  [`duplicate-flags.csv`](duplicate-flags.csv) and [`audit-summary.json`](audit-summary.json).

`conditions.csv` is filled in **by hand**. If it does not exist, the audit creates a template
with one blank row per photo. New photos get blank rows appended, and filled-in rows are never
changed. Rerun the audit after every edit. The evaluation refuses to run if the manifest was
edited after the audit, if any photo changed, or if any included photo has a blank or invalid
condition.

Flagged duplicates and GPS photos should be **removed or retaken** before the evaluation, so
that the final set is clean rather than silently filtered.

## Counts

Filled in from `audit-summary.json` after the final audit.

| Class | Photos | Included |
| --- | ---: | ---: |
| cardboard | TODO | TODO |
| glass | TODO | TODO |
| metal | TODO | TODO |
| paper | TODO | TODO |
| plastic | TODO | TODO |
| trash | TODO | TODO |
| **total** | TODO | TODO |

| Condition | Values (included photos) |
| --- | --- |
| background | TODO |
| lighting | TODO |
| device | TODO |

## How it is used

```sh
waste-phase2-eval robustness --output docs/phase-2/robustness
```

This command runs **once**. It scores every included photo with Phase 1 SVM, A2.1 CNN and H3
Hybrid through `predict_phase2`, which applies the frozen temperatures and demo thresholds. It
saves `robustness-predictions.csv` and refuses to run again once that file exists.
`waste-phase2-eval robustness-report` rebuilds the tables from the saved predictions without
scoring anything.

## Limitations

- **Small.** About 300 photos and about 50 per class, so confidence intervals are wide and
  per-condition groups are smaller still.
- **Collected by students.** The photos come from our own homes and campus with our own phones.
  They are not a random sample of real waste streams. Backgrounds, lighting and phones were not
  fully crossed, so these conditions are correlated.
- **Visual material labels.** Each label is the main visible material, chosen by the
  photographer, in TrashNet's six classes. It says nothing about local recycling rules. The
  `trash` class stays a catch-all, as in TrashNet.
- **Not a training set.** If the results show a large drop, closing that gap needs new training
  data *and* a new untouched external set. These photos cannot be both.
