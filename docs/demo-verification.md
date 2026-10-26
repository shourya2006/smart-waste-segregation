# Demo runtime verification

Verified locally on 2026-10-26 by Shourya Bafna, following Saksham Miglani's
demo and submission contribution in PR #8. The merge preserves Saksham's
original commit and authorship; runtime corrections are a separate contribution.

## Runtime corrections

- Removed the repository's substitute `gradio` module, which shadowed the
  installed library and did not start a real server.
- Use Gradio 6.30.0, with CSS supplied to `launch()` for that version.
- Connect Clear to the uploaded image and all six result outputs; show errors
  in the visible Status field. Image changes clear previous results.
- Allow the confidence message enough space on narrow screens and hide empty
  uncertainty banners.
- Remove import shims that shadowed the installed project package.
- Install the actual demo dependencies in CI and run lint across the repository.

## Environment and artifact

Local verification used Python 3.12.5 and a fresh environment installed from
`requirements-demo.lock`. `pip check` reports no broken requirements.
The demo pins Pillow 12.3.0 because Gradio requires Pillow below 13.
`requirements.lock` retains historical evaluation Pillow 13.0.0; all other
base pins match the demo lock. This is an explicit environment difference.

The original evaluated artifact remains outside Git:

- Path: `artifacts/calibrated_svm/model.joblib`
- Size: 66,270,283 bytes
- SHA-256: `83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112`

A bounded compatibility check compared six generated inputs (RGB PNG,
random RGB PNG/JPEG, grayscale, RGBA and EXIF-oriented JPEG) under Pillow
13.0.0 and 12.3.0. All 2,760-dimensional feature vectors were bit-for-bit
identical and prediction fields matched. This does not establish equivalence
for every possible image or repeat the held-out performance evaluation.

## Checks and browser evidence

- Full suite: **162 tests passed, zero skipped**, with the original artifact present.
- Without the artifact, the one real-artifact test is expected to skip; synthetic
  HTTP integration checks still run with the installed Gradio server.
- Repository-wide `ruff check .` and dependency consistency checks passed.
- New HTTP integration checks start a real server, upload a generated image,
  exercise prediction and empty-input errors, and verify Clear's seven outputs.
- Chromium checks exercised upload, classification, empty-input feedback and
  clearing. Desktop viewport: 1280 x 800. Mobile viewport: 390 x 844, with no
  horizontal page overflow.
- The screenshot input is a **generated solid-color patch**, classified by the
  evaluated model as Cardboard at 53.9%, with six probabilities and the frozen
  threshold 0.52. It is a functional check, not evidence of classification quality.
- After server startup, Chromium recorded no page errors during upload and
  prediction; two Gradio frontend warnings said "Method not implemented."

The full-page captures in [demo-desktop.png](demo-desktop.png) and
[demo-mobile.png](demo-mobile.png) show this real-model session. They were
captured with browser automation after prediction completed. The standalone
screenshot helper captures the initial interface instead.

## Preservation and remaining work

No model fitting, calibration, threshold selection or final test evaluation
was rerun. The model checksum and tracked `docs/evaluation/` records are
unchanged. Historical evaluation source reconstruction remains limited as
described in the evaluation documentation.

The [LaTeX report PDF](../output/pdf/phase-1-report.pdf) has been compiled and
visually verified. The earlier HTML report is an archived preview.
A teammate should reproduce the demo from a fresh checkout with the original
artifact supplied separately; the isolated local dependency check does not
replace that handoff.
