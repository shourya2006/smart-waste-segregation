# Shared training protocol

Shourya's implementation provides the common inputs, folds and experiment schema
for both model owners. The pinned Python environment in `requirements.lock` is
required. No validation/test predictions or final performance estimates are made
by these commands.

## Reproduce

From the repository root, with the environment activated:

```sh
python -m pip install --no-deps -e .
waste-data download
waste-data features
waste-train folds
waste-train svm --output artifacts/my-svm-run
```

Feature generation uses the existing audited manifest. Do not run audit/split
again merely to start training. Relative CLI paths resolve against `--root`.
Global options such as `--cache`, `--config` and `--folds` go before the subcommand.
For example:

```sh
waste-train --root /path/to/smart-waste-segregation folds
waste-train svm --models linear_svm --descriptors color hog --output artifacts/color-hog
```

Every output experiment directory must be new. Existing fold files are validated
and never silently replaced. A changed seed, training cache, class mapping,
manifest, feature source or scikit-learn version requires an explicit, documented
protocol decision rather than overwriting the shared assignments.

## Fixed folds and model inputs

`folds.json` maps each of the 1,764 outer-training image IDs to one of five
validation folds. It contains no images, feature vectors or model weights.
Assignments use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
Each row is held out once. Related images stay together, and every fold contains
all six classes. Fold indices address only the arrays returned by the loader;
they do not address the full 2,521-row feature cache.

```python
from pathlib import Path
from waste_sorting.training import (
    load_training_data, load_or_create_folds, feature_columns, run_experiment,
)

root = Path.cwd()
data = load_training_data(root / "data/processed/features.npz", root)
fold_path = root / "docs/training/folds.json"
cv = load_or_create_folds(data, fold_path, n_splits=5, seed=42)
X_train, y_train, groups_train = data.X, data.y, data.groups
columns = feature_columns(data, ["color", "hog", "lbp"])
```

The loader checks the current cache metadata and manifest, exact included image
IDs/order, labels, integer class mapping, split assignments, duplicate groups,
feature names/boundaries, shape, dtype and finite values. It rejects groups that
cross outer splits, then returns read-only training arrays. The fingerprint
includes training vector bytes and row identities as well as source/configuration
provenance. All model/descriptor studies use this same full-data fingerprint.

## Daksh's Random Forest handoff

Use the shared loader and `cv` above. Keep the full cached feature vector as the
input to the estimator; select `columns` inside a scikit-learn pipeline using a
`ColumnTransformer`. This keeps feature selection consistent when the saved
pipeline receives new feature vectors. Do not manually prefit any learned
preprocessing outside CV.

Call `run_experiment` with your pipeline, grid, model name, descriptor names,
fresh output directory, repository root, `fold_path` and seed 42. The runner
checks that supplied folds match the saved file, applies six-class macro F1
with `zero_division=0`, searches training folds, and refits the winner on outer
training rows only. It works with scikit-learn-compatible estimators and uses
the same result schema for Random Forest and SVM. Keep estimator parallelism
and search parallelism coordinated to avoid excessive worker counts.

## SVM choices and outputs

The default study uses all 1,806 unscaled cached features. Parameters are
predeclared in `configs/phase1.toml`:

- Dummy: predict the most frequent training class (one candidate).
- Linear: `LinearSVC(dual="auto", max_iter=20000, random_state=42)`;
  C in 0.1, 1, 10; class weight none/balanced (six candidates).
- RBF: `SVC(kernel="rbf", random_state=42)`;
  the same C/weight options; gamma in scale, 0.001, 0.01 (18 candidates).

Both SVM pipelines contain `StandardScaler` after feature selection. Grid search
clones and fits the complete pipeline in every training fold. No global scaler,
PCA, data augmentation, calibration or probability fitting is used here. With
the default single-worker search, convergence warnings abort the experiment;
the default linear iteration limit must be adequate before publishing results.

Each model run saves locally under its output directory:

- `cv_results.csv`: every candidate's parameters, five macro F1 scores, mean/std,
  rank and fit/scoring timing.
- `summary.json`: winning parameters and scores, feature dimensions, seed,
  fold-file checksum, manifest/feature/config/training fingerprints, source
  snapshot, Git revision and dirty-state flag, environment and runtime.
- `estimator.joblib`: best full pipeline refitted on all outer training rows.
- `failure.json` if fitting fails, rather than publishing a successful result.

Model binaries and full run outputs stay in ignored `artifacts/`. The small,
reviewable fold assignments and verified result summary belong in `docs/training/`.
Runs before publication record the base Git revision plus a dirty-state flag and
the exact Python source-content checksum; they are not attributed to a future
commit that does not exist yet.

CV scores were used to choose parameters, so they are selection scores rather
than unbiased estimates of generalization. Prajjwal must later clone the chosen
complete pipeline for group-aware held-out-training calibration, choose the
abstention threshold on the outer validation split and evaluate the frozen
system on the untouched test split. These saved estimators are uncalibrated.

## Verified initial results

Real-data runs on 23 October 2026 used all 1,806 features, 1,764 training rows
and 1,729 duplicate groups. Validation-fold sizes are 354, 354, 352, 352 and 352.
All 25 candidates were evaluated on the same five folds (125 CV fits plus
three final outer-training refits). No outer validation/test predictions were
generated. The single-worker command also set `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` for the recorded run.

| Model | Mean CV macro F1 | Fold std | Selected settings | Search/refit seconds |
| --- | ---: | ---: | --- | ---: |
| Most-frequent dummy | 0.0636 | 0.0002 | Most frequent class | 0.05 |
| Linear SVM | 0.6007 | 0.0284 | C=0.1, no class weighting | 173.81 |
| RBF SVM | 0.6731 | 0.0339 | C=10, gamma=scale, no class weighting | 143.82 |

RBF is strongest among these three initial baselines; Random Forest and the
planned feature-family studies are still pending. Runtime is machine-specific.
The standard deviations are fold variation, not confidence intervals.

The [run summaries](svm-cv-summary.json) preserve full precision, provenance and
winning parameters; the [candidate table](svm-cv-results.csv) contains all 25
candidates and their individual fold scores. All saved models were reloaded,
their class/input dimensions checked, and both SVM scalers' fitted means/sample
counts matched all outer-training rows. Source snapshots matched the current
implementation. Local package consistency and all 68 tests passed.

## Checks

```sh
ruff check .
python -m unittest discover -s tests -v
```

Training tests include deliberately extreme held-out vectors, misaligned cache
identities/targets, excluded rows, non-finite/wrong-sized matrices, outer-group
leakage, exhaustive group-aware folds, stale/edited fold files, and observed
scaler fitting inputs in every CV fold plus the final outer-training refit.
