# Om Yadav preprocessing and data verification handoff

Prepared on 23 October 2026 with the coding assistant for Om's assigned component.
This records assistant-assisted implementation and verification, not an independent
human review. Publication on the preprocessing branch was approved by the user.

## Implemented interface

`from waste_sorting.preprocessing import preprocess_image`

`preprocess_image(image, config)` accepts a PIL image and the complete mapping
loaded from `configs/phase1.toml`. It returns a new RGB PIL image at the configured
width and height. Aditya should call this function before descriptor extraction;
the future prediction path should reuse it as well.

The operation corrects EXIF orientation without modifying the input's pixels or
orientation metadata. Grayscale and palette images are supported. Alpha is
composited onto `padding_rgb` before resizing to prevent transparent regions
becoming unintended dark objects.

## Resizing policy and rationale

Use the smaller of target-width/source-width and target-height/source-height as
the scale factor. Resize with LANCZOS and allow upscaling. Round dimensions to
the nearest integer using Python's ties-to-even rule, with each dimension clamped
between one and its canvas dimension. Center with floor division; an odd padding
pixel goes on the right or bottom. This preserves shape apart from unavoidable
integer rounding and avoids stretching objects before HOG or texture extraction.

With current defaults, a 512x384 image becomes 128x96 with 16 white rows above
and below. The same deterministic operation must run in training and inference.
No augmentation, fitted normalization, or statistical learning occurs here.
Full-image descriptors will include padding/background; this is a limitation
to examine during feature comparisons and external-image evaluation.

Invalid input types, missing/invalid configuration, non-positive dimensions,
invalid RGB channels, and requests to stretch the image are rejected. Decode
failures are reported with a clear error retaining the underlying exception.

## Reverification of existing data

The pinned archive SHA-256 was checked. The audit, candidate generation, review
application, and seed-42 group-aware split assignment were recomputed in memory.
The resulting manifest reproduced the committed SHA-256 exactly:

`affd3e9d29e612b0179820a2a20ed21395e26d89a094f02bb7390d9400563314`

All 2,527 source images remain readable RGB images at 512x384. The three
conflicting-content groups still exclude six copies, leaving 2,521 usable images.
The unchanged split counts are 1,764 training, 380 validation, and 377 test.
All 45 existing pHash candidates and their review decisions were preserved.

## Second inspection of the six uncertain pairs

The assistant inspected new, larger local contact sheets of these pairs. There
is insufficient evidence to resolve item identity confidently, so the existing
conservative groups are retained. This does not establish exhaustive same-item
detection beyond the candidate-generation method.

| Pair | Observation and retained decision |
| --- | --- |
| cardboard15 / cardboard361 | Similar dark ridged surface and diagonal boundary; framing and lighting differ. Retain uncertain. |
| cardboard386 / cardboard72 | Matching general texture without a unique identifying feature. Retain uncertain. |
| metal149 / metal9 | Same purple can branding, but different deformation. Retain uncertain. |
| metal155 / metal6 | Circular views could show opposite ends of one can; identity is not established. Retain uncertain. |
| metal165 / metal166 | Different foil silhouettes could be one reshaped piece or separate pieces. Retain uncertain. |
| plastic153 / plastic9 | Same packaging type with different crushing; no unique identifier proves identity. Retain uncertain. |

No approved dataset CSV, source image, exclusion, or split was edited. The local
review sheets remain outside Git under `work/om-verification/`.

## Verification and reproduction

Run from the repository root after the documented environment setup:

```sh
ruff check .
python -m unittest discover -s tests -v
```

The suite passes 32 tests: the 19 existing data tests and 13 new preprocessing
tests. New checks cover real project configuration, portrait/landscape geometry,
odd padding, grayscale, partial/full transparency, palette transparency, EXIF,
input preservation, deterministic independent outputs, very thin images,
invalid arguments, stretching rejection, and closed lazy-image decode failure.

Representative training images from all six classes were also processed for
visual inspection. No classifier was trained and no test prediction scores were
computed. Feature extraction, classifiers, and calibration remain other members'
components.

## Viva topics

Explain why resizing must preserve shape for HOG; what letterboxing adds to the
image; why alpha must be composited; how EXIF orientation differs from pixel
dimensions; why preprocessing must be identical at inference; why deterministic
image transforms can run before splitting while learned statistics cannot; and
why conservative duplicate grouping reduces an avoidable evaluation shortcut.
