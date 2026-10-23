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
and linear/RBF SVM searches with fold-local scaling. See the
[training guide](docs/training/README.md) for reproduction and teammate interfaces.
Random Forest/feature studies, calibration, final evaluation and the demo follow.
Training-CV selection scores must not be interpreted as final-test performance.

## Local setup

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
ruff check .
python -m unittest discover -s tests -v
python -c "import waste_sorting; print(waste_sorting.__version__)"
```

The upload demo will use the optional dependencies installed with
`python -m pip install -e '.[dev,demo]'` once it is implemented.
Dependency ranges are declared in `pyproject.toml`. `requirements.lock` pins the
resolved Python 3.12 development environment, excluding the optional demo.

## Repository layout

```text
configs/                  Initial experiment settings
data/                     Acquisition and provenance instructions
  raw/                    Local source images, ignored by Git
  processed/              Local manifests and feature caches, ignored by Git
src/waste_sorting/        Reusable Python package
notebooks/                Exploratory analysis
artifacts/                Local trained models and generated figures, ignored by Git
docs/                     Phase 1 milestones and contribution records
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
