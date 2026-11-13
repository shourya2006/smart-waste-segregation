# Phase 2 reproduction and audit guide

## Scope

This guide documents **read-only** reproduction of the frozen Phase 2 results. Do **not** rerun model selection on the test set or modify `docs/phase-2/evaluation/protocol.json`. Test predictions and metrics were produced once by the team and are included in the repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,demo,dl]'
python -m pip install pytest jupyter nbformat
```

Run commands from the repository root. For notebook analysis: `jupyter notebook notebooks/phase2-summary.ipynb`.

## Frozen evidence

- `docs/phase-2/evaluation/protocol.json`: frozen models, temperatures, demo/headline thresholds, file hashes and provenance.
- `docs/phase-2/evaluation/test-metrics.json`: final metrics, uncertainty, statistical comparisons.
- `docs/phase-2/evaluation/test-predictions.csv`: saved per-image predictions.
- `docs/phase-2/evaluation/README.md`: detailed evaluation and limitations.
- `report/phase2.tex` and `output/pdf/phase-2-report.pdf`: IEEE-style paper and compiled PDF.
- `presentation/phase2-slides.pptx` and `output/pdf/phase-2-presentation.pdf`: editable and PDF slide deck.

## Frozen results (N = 377)

| Model | Test macro F1 | Test accuracy | Demo threshold |
|---|---:|---:|---:|
| Phase 1 refit SVM | 0.8963 | 90.19% | 0.76 |
| A2.1 CNN | 0.8577 | 88.06% | 0.87 |
| H3 late-fusion hybrid | 0.9233 | 93.63% | 0.53 |

The historical Phase 1 ensemble had macro F1 0.8999; it is not the Phase 2 refit SVM. H3 vs A2.1 McNemar p ≈ 0.000019 survives Bonferroni α = 0.0167. H3 vs SVM p ≈ 0.0351 does **not** survive correction.

## Demo model artifacts

The large trained model files are **not** distributed in this PR. Place the team's frozen, unmodified artifacts at the paths recorded in `docs/phase-2/evaluation/protocol.json` and `src/waste_sorting/phase2_demo.py`:

- `artifacts/phase2/final/A2.1/model.pt` and `metadata.json`
- `artifacts/phase2/hybrid-final/h3_late_fusion/svm_pipeline.joblib` and `metadata.json`

The demo checks SHA-256 hashes and refuses mismatched artifacts. Never bypass hash verification or commit large weights. Confirm actual archive layout after extracting team-supplied model ZIPs.

```bash
python -m pytest -q tests/test_phase2_demo.py
python -m ruff check app.py src/waste_sorting/phase2_demo.py tests/test_phase2_demo.py
python app.py --help
```

With artifacts present, launch the app using the arguments shown by `python app.py --help`, select each Phase 2 model, upload an image, and capture screenshots showing probabilities and the uncertainty banner. **Real-model screenshot and validation consistency checks remain pending until artifacts are present.**

## IEEE report compilation

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error phase2.tex
bibtex phase2
pdflatex -interaction=nonstopmode -halt-on-error phase2.tex
pdflatex -interaction=nonstopmode -halt-on-error phase2.tex
```

## Limitations

All test results are on TrashNet with white backgrounds; no out-of-distribution robustness dataset was collected. The `trash` class has only 20 test examples. Model-selection reuse of CV folds may inflate CV estimates. Grad-CAM uses a heuristic foreground mask on 12 images and cannot rule out background dependence.
