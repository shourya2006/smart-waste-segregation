# Smart Waste Segregation Assistant

Project 12: an educational image classifier for cardboard, glass, metal, paper,
plastic, and the TrashNet dataset's miscellaneous trash category.

Phase 1 uses handcrafted color histograms, Histogram of Oriented Gradients
(HOG), and Local Binary Patterns (LBP), compared with SVM and Random Forest.
Transfer learning with MobileNetV2 or EfficientNet is reserved for Phase 2.

## Current status

Dataset preparation is complete: 2,527 images audited, six images with conflicting
labels excluded, 45 near-duplicate candidates visually reviewed, and 2,521 usable
images assigned to fixed group-aware splits. Training contains 1,764 images,
validation 380, and test 377. Training-only EDA is available in
[the dataset report](docs/dataset/README.md).

Shared preprocessing and the 1,806-dimensional HSV/HOG/LBP extractor are implemented.
The training protocol provides five fixed group-aware folds, a dummy baseline,
and linear/RBF SVM searches with fold-local scaling. Random Forest studies compare
five descriptor families and class weighting on the same folds. See the
[training guide](docs/training/README.md) for reproduction and teammate interfaces.

The extended RBF SVM model (0.8803 training-CV macro F1) was selected and calibrated
using group-aware Platt scaling. Abstention was selected on validation data (threshold 0.52).
Frozen test evaluation on 377 images achieved 0.8999 macro F1 and 90.19% unconditional
accuracy (91.37% accepted accuracy at 98.41% coverage).

The interactive Gradio demo (`app.py`), integration tests, reproducible demo lockfile,
IEEE technical report sources (`report/`), and presentation sources (`presentation/`) are prepared.
The [compiled IEEE report](output/pdf/phase-1-report.pdf),
[editable presentation](output/presentation/smart-waste-segregation-final.pptx), and
[presentation PDF](output/pdf/project-presentation.pdf) are ready. The earlier HTML
report is archived and does not contain every correction in the final PDF.

## Local setup

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# To install base and development tools:
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .

# OR to install the full environment including the Gradio demo:
python -m pip install -r requirements-demo.lock
python -m pip install --no-deps -e .

ruff check .
python -m unittest discover -s tests -v
```

`requirements.lock` pins the base model-runtime environment. `requirements-demo.lock` locks the reproducible demo environment (Gradio 6.30.0 and dependencies) with exact dependency pins. The sole base-version difference is Pillow 12.3.0 in the demo, required by Gradio's Pillow <13 constraint; the original evaluation environment remains Pillow 13.0.0 in `requirements.lock`. See [demo verification](docs/demo-verification.md) for the bounded compatibility check.

## Running the Photo Upload Demo

The demo runs locally using Gradio and reuses the shared prediction interface (`predict_image`).

### 1. Model Artifact Placement & Verification

The evaluated model binary is excluded from Git to keep the repository lightweight:
- **Expected location:** `artifacts/calibrated_svm/model.joblib`
- **Expected SHA-256:** `83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112`
- **Expected file size:** `66,270,283` bytes (~66.3 MB)

Verify the binary integrity before running:
```sh
python -c "
import hashlib, pathlib
p = pathlib.Path('artifacts/calibrated_svm/model.joblib')
assert p.exists(), 'Model artifact missing'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112', 'Checksum mismatch'
print('Model artifact verified.')
"
```

### 2. Launching the Demo

```sh
# Launch with the preserved evaluated model:
python app.py --port 7860

# Or launch with synthetic fixtures (if model binary is unavailable):
python app.py --synthetic-fixture --port 7860
```
Open `http://127.0.0.1:7860` in your browser.

### 3. Demo Features & Safety Behavior
- **In-Memory Model Reuse:** The 66 MB model binary is validated and loaded once per process and reused across requests.
- **Supported Inputs:** JPEG, PNG, or standard still-image formats showing a single waste item. Images exceeding 8,192 pixels on any side or 16,777,216 pixels total are rejected before feature extraction. Gradio decodes browser uploads before the application callback.
- **Uncertainty & Abstention:** Predictions with calibrated maximum probability below the frozen threshold (**0.52**) prominently display **"Uncertain — manual inspection recommended"**. A tentative category is displayed separately without endorsement.
- **Stale State Clearing:** Predictions are instantly wiped whenever an uploaded photo is cleared or replaced.
- **Privacy & Security:** Public sharing is disabled (`--share` is rejected). The application does not save uploaded photos as dataset or project files. Gradio creates temporary upload/cache files in system temporary storage; those files may remain after a browser session.

## Reproducing Reported Metrics vs. Running the Demo

Neither task requires refitting models or rerunning final test evaluation:

1. **Reproduce reported evaluation metrics (no model refitting):**
   Committed records in `docs/evaluation/` serve as the source of truth:
   ```sh
   # Verify test set metrics from saved sample-by-sample predictions:
   python -c "
   import pandas as pd
   df = pd.read_csv('docs/evaluation/test-predictions.csv')
   print('Test samples:', len(df))
   print('Unconditional accuracy:', f'{df[\"correct\"].mean()*100:.2f}%')
   acc = df[~df['abstained']]
   print(f'Selective accuracy: {acc[\"correct\"].mean()*100:.2f}% at {len(acc)/len(df)*100:.2f}% coverage')
   "
   ```
2. **Run the local demo (requires model binary):**
   Loads the preserved `artifacts/calibrated_svm/model.joblib` to classify new single-item uploads without retraining.

## Repository layout

```text
configs/                  Initial experiment settings
data/                     Acquisition and provenance instructions
  raw/                    Local source images, ignored by Git
  processed/              Local manifests and feature caches, ignored by Git
src/waste_sorting/        Reusable Python package
notebooks/                Exploratory analysis
artifacts/                Local trained models and generated figures, ignored by Git
docs/                     Phase 1 milestones, evaluation records, and member notes
report/                   LaTeX/IEEE report project, HTML export, and figures
presentation/             Editable presentation sources and interactive slide viewer
requirements.lock         Pinned base model-runtime environment
requirements-demo.lock    Pinned demo environment (Gradio 6.30.0 + base dependencies)
app.py                    Local photo upload demo entry point
.github/workflows/        Automated repository checks
```

## Reproduce dataset preparation

From the repository root, after environment setup:

```sh
waste-data download
waste-data audit
waste-data split
waste-data eda
```

The download is pinned to a source revision and SHA-256 checksum. The committed
review decisions apply to this archive at pHash distance 6. Audit regenerates
candidates and local review sheets; split refuses missing or stale reviews and
changed source files. If changing the data or candidate threshold, review the
new pairs before replacing split assignments. Do not change frozen splits after
model evaluation begins. Keep uncertain related images in the same partition.

The [tracked manifest](docs/dataset/manifest.csv) includes every source image,
exclusion reason, hashes, duplicate group, and assigned split. Perceptual hashing
does not establish exhaustive same-item detection; this limitation is documented.

## Reproduce training

After the pinned environment is installed and raw images are downloaded:

```sh
waste-data features
waste-train folds
waste-train svm --output artifacts/my-svm-run
```

The loader returns only the outer training split. Both model owners use the
committed [fold assignments](docs/training/folds.json) and shared experiment
runner. Models/caches stay local; choose a new output directory for each run.
The [training guide](docs/training/README.md) explains descriptor subsets,
fingerprints, result files and the Random Forest handoff.

## Evaluation commitments

- Select feature sets and model parameters using training cross-validation.
- Fit scaling and other learned preprocessing within each training fold.
- Fit confidence calibration using held-out training folds.
- Select the low-confidence abstention threshold using validation data.
- Freeze the pipeline before final test evaluation.
- Report macro F1, per-class recall, confidence calibration, and failure cases.
- Report full six-class results separately from results after abstention.

The implementation roadmap is in [docs/phase-1-plan.md](docs/phase-1-plan.md).
Team workflow is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Dataset and references

Use the [official TrashNet repository](https://github.com/garythung/trashnet)
and acknowledge its authors. See [data/README.md](data/README.md) for acquisition
and provenance requirements. The course-provided instructions and project PDF
are not redistributed in this repository.

## Limitations

This educational model will predict visual dataset categories. Local recycling
rules, contamination, mixed materials, lighting, and background affect actual
disposal decisions and prediction reliability. The trash category is not a
general unknown-object detector.
