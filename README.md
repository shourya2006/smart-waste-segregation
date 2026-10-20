# Smart Waste Segregation Assistant

Project 12: an educational image classifier for cardboard, glass, metal, paper,
plastic, and the TrashNet dataset's miscellaneous trash category.

Phase 1 uses handcrafted color histograms, Histogram of Oriented Gradients
(HOG), and Local Binary Patterns (LBP), compared with SVM and Random Forest.
Transfer learning with MobileNetV2 or EfficientNet is reserved for Phase 2.

## Current status

Repository foundation only. Dataset preparation, feature extraction, training,
evaluation, and the photo upload demo have not been implemented yet. No accuracy
or other experimental results are claimed.

## Local setup

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
python -c "import waste_sorting; print(waste_sorting.__version__)"
```

The upload demo will use the optional dependencies installed with
`python -m pip install -e '.[dev,demo]'` once it is implemented.
Dependency ranges are declared in `pyproject.toml`; an exact environment lock
will be committed after validating the first working data pipeline.

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

## First implementation milestone

Acquire TrashNet, validate every image, inventory class counts, identify exact
and near duplicates, and create a reproducible image manifest. Review duplicate
groups before assigning approximately 70/15/15 train/validation/test splits.
No related image group may cross split boundaries.

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
