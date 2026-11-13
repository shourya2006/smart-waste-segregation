# Phase 2 reproduction guide

How to reproduce every Phase 2 result from a fresh clone. Run all commands from the
repository root with the virtual environment active.

There are two levels:

- **Check the frozen results.** No training is needed (Sections 1 to 3, about 15 minutes).
  Use this before the viva.
- **Retrain from scratch.** This rebuilds every number (Section 4, several hours on a GPU).

## 1. Setup

Use Python 3.12. On an Apple-silicon Mac, first check that
`python3.12 -c "import platform; print(platform.machine())"` prints `arm64`. If it prints
`x86_64`, PyTorch will not install; create the environment with
`uv venv --python cpython-3.12-macos-aarch64 .venv` instead of the first line below.

```sh
git clone https://github.com/shourya2006/smart-waste-segregation.git
cd smart-waste-segregation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.lock
python -m pip install -r requirements-dl.lock   # Linux without an NVIDIA GPU: add --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
ruff check .
python -m unittest discover -s tests -v
```

The locks pin every package version used for the reported results. The tests use the
standard library's `unittest`; no other test runner is needed.

## 2. Data

```sh
waste-data download
waste-dl-data        # must print train 1764, validation 380, test 377
```

The download is pinned to a source revision and checksum. Never run `waste-data split`:
the splits are frozen in `docs/dataset/manifest.csv`.

## 3. Check the frozen results (no training)

### 3.1 Get the two frozen models

The model files are kept out of Git. Get both zips from the team Drive and unzip them so the
files land at these paths:

| Zip | Unzip to | File and SHA-256 |
| --- | --- | --- |
| `A2.1-final-model.zip` | `artifacts/phase2/final/A2.1/` | `model.pt`: `26592f8b5aee95eda28ef6088af8c7d737e7e6b9d38a6733c9faa5c1b4c5d28e` |
| `h3_late_fusion.zip` | `artifacts/phase2/hybrid-final/h3_late_fusion/` | `svm_pipeline.joblib`: `ca5a5411db8fbbe79296e552e50f7340fe3c3763148697151a83cc858f5d080a` |

Check them. The loaders verify every hash and refuse a mismatched file:

```sh
python -c "from waste_sorting.hybrid import load_hybrid; h = load_hybrid('artifacts/phase2/hybrid-final/h3_late_fusion'); print(h.w, h.metadata['a2_model_sha256'])"
```

It must print `0.5 26592f8b5aee95eda28ef6088af8c7d737e7e6b9d38a6733c9faa5c1b4c5d28e`.

### 3.2 Recompute the reported numbers

- `notebooks/phase2-summary.ipynb` rebuilds the main tables from the committed result files
  (`test-metrics.json`, `protocol.json`, `docs/phase-2/ablations/summary.csv`). It needs no
  models and does no training. Run it with `jupyter notebook`, or read it on GitHub, where the
  outputs are saved.
- `docs/phase-2/evaluation/test-predictions.csv` holds every test prediction. All test
  metrics, bootstrap intervals and McNemar tests are computed from this file.

### 3.3 Run the demo

```sh
python app.py --port 7860
```

Open `http://127.0.0.1:7860`. The Phase 2 model selector reads each model's temperature and
demo threshold from `docs/phase-2/evaluation/protocol.json`. It refuses to start a model whose
files do not match the frozen hashes. `--share` is rejected on purpose.

## 4. Retrain from scratch

Each step writes to a new output folder and refuses to overwrite an earlier run. Commands
are listed in the order the results were produced.

| Step | Command | Time (Apple M-series GPU) | Repeatable exactly? |
| --- | --- | --- | --- |
| Frozen-backbone embeddings | `waste-dl-embeddings --backbone mobilenet_v2 --splits train` (same for `efficientnet_b0`) | ~1 min each | yes on CPU; tiny GPU differences |
| 13 pre-registered ablations | `caffeinate -i python scripts/run_ablations.py`, then `python scripts/run_ablations.py summarise` | 5-10 h (resumable) | no: GPU training is not bit-identical |
| Final A2.1 CNN | `waste-dl-train final --epochs 13 --output artifacts/phase2/final/A2.1 --set dl.training.unfreeze_blocks=2` | ~1-2 min | no (GPU); a retrained model gets a new hash |
| Phase 1 SVM out-of-fold probabilities | `waste-hybrid phase1-oof --output artifacts/phase2/hybrid/phase1_svm_oof.csv` | ~1 min (CPU) | yes |
| H1/H2 hybrid candidates | `waste-hybrid cv --candidate <name> --family <h1 or h2> --backbone <backbone> --classifier <classifier> --C <C> --gamma <gamma> --output artifacts/phase2/hybrid/<name>`, one run per row in `docs/phase-2/hybrid/summary.csv` | minutes each (CPU) | yes |
| H3 late fusion (nested CV) | `waste-hybrid late-fusion --output artifacts/phase2/eval/h3 --svm-oof artifacts/phase2/hybrid/phase1_svm_oof.csv` | seconds | yes |
| Final fitted hybrid | `python -c "from waste_sorting.hybrid import fit_final_hybrid; fit_final_hybrid()"` (uses the A2.1 model above, with `w = 0.5`) | ~1 min (CPU) | yes, for the same A2.1 model |
| Calibration and frozen protocol | `waste-phase2-eval calibrate --output docs/phase-2/evaluation --h3-oof artifacts/phase2/eval/h3/oof_predictions.csv --phase1-oof artifacts/phase2/hybrid/phase1_svm_oof.csv` | ~2 min | yes, for the same models |
| Single test evaluation | `waste-phase2-eval test --output docs/phase-2/evaluation` | ~2 min | yes, for the same models |

Notes:

- `waste-phase2-eval calibrate` refuses to overwrite an existing `protocol.json`, and
  `waste-phase2-eval test` refuses to run again once `test-predictions.csv` exists. To repeat
  the evaluation, write to a new `--output` folder. Never edit the committed protocol or the
  committed test results.
- A retrained A2.1 model gets a different hash, so the committed `protocol.json` will (correctly)
  refuse it. A full retrain therefore yields new, separately frozen results.
- Use one device for all runs in a comparison. GPU and Apple MPS arithmetic is not bit-for-bit
  repeatable, so runs on different devices can differ slightly.

## 5. Build the report

Needs a LaTeX installation (`pdflatex`, `bibtex`):

```sh
make -C report phase2
cp report/phase2.pdf output/pdf/phase-2-report.pdf
```

## 6. Known limitations

- **CV scores are somewhat optimistic,** because configurations were chosen on the same five
  folds. The single held-out test evaluation is the unbiased estimate.
- **The test set is small:** 377 images, of which 20 are trash.
- **Everything is from one domain.** TrashNet photos were taken against a white posterboard, and
  the planned set of everyday photos was not collected.
- **The final hybrid was fitted from an uncommitted working tree.** Its metadata shows
  `working_tree_dirty: true`. Its file hashes are pinned in `protocol.json`, so the model that was
  evaluated cannot change.
