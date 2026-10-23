# Daksh Batra: Random Forest, Feature Studies, and Class Weighting

Prepared on 23 October 2026 with the coding assistant for Daksh's assigned Phase 1 component.
Code, tests, and experiments are assistant-assisted. The team remains responsible for understanding
and explaining all methods for the individual viva.

## Implemented Contribution

- `src/waste_sorting/models/random_forest.py`:
  - Reusable scikit-learn `Pipeline` builder encapsulating internal descriptor column selection via
    `ColumnTransformer` and `RandomForestClassifier`.
  - Invariance to monotonic feature scale: no `StandardScaler` is required or included.
  - Predeclared parameter grid parser reading candidate values from configuration (`configs/phase1.toml`).
- `src/waste_sorting/training.py`:
  - Extended CLI with the `rf` subcommand to search Random Forest models and feature studies across shared folds.
  - Coordinated worker concurrency (`--n-jobs` for search, `--estimator-n-jobs` for estimator) to prevent thread oversubscription.
- `tests/test_random_forest.py`:
  - 4 comprehensive unit and integration tests covering:
    - Pipeline column selection and feature dimensions (HSV 32, HOG 1,764, LBP 10, HSV+HOG 1,796, Combined 1,806).
    - Confirmation that feature scaling (`StandardScaler`) is absent from the Random Forest pipeline.
    - Parameter grid parsing and error handling for malformed configs.
    - Experiment execution reproducibility, result logging, and verification that saved estimators accept full 1,806-dim feature vectors.
    - CLI execution with artifact and CSV generation.
- `docs/training/rf-cv-summary.json`:
  - Machine-readable summary preserving winning parameters, per-fold scores, fold standard deviation, dimensions, timings, fingerprints, environment, and provenance for all 5 feature studies.
- `docs/training/rf-cv-results.csv`:
  - Complete record of all 120 evaluated candidates across the 5 feature sets (24 candidates per feature set).

## Statistical Protocol & Experimental Setup

1. **Outer Split Protection**:
   - The outer dataset split contains 1,764 training images, 380 validation images, and 377 test images.
   - The shared loader (`load_training_data`) validates manifest integrity and duplicate group isolation, then discards validation and test images.
   - Zero outer validation or test data was used for model training or hyperparameter selection.
2. **Fixed Group-Aware Cross-Validation**:
   - Evaluated across the exact 5 shared folds defined in `docs/training/folds.json` (`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`).
   - Duplicate image groups stay strictly within the same fold to prevent data leakage.
   - Validation-fold sizes are 354, 354, 352, 352, and 352.
3. **Metric**:
   - Six-class macro F1 (`f1_macro` with `zero_division=0`), treating all classes equally (preventing the majority paper class from dominating minority trash).
   - Mean CV macro F1 across the 5 folds determines model selection.
   - Reported standard deviations reflect fold-to-fold variation, not confidence intervals.
   - These are model selection scores and should not be confused with final unbiased test generalization.

## Random Forest Theory & Implementation Decisions

- **Ensemble Decision Trees & Bagging**:
  - Random Forest builds an ensemble of decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspace sampling at each split ($m = \sqrt{p}$).
  - Gini impurity is used to select optimal orthogonal feature splits.
  - Because decision tree splits depend only on the rank ordering of feature values along individual axes, Random Forest is strictly invariant to monotonic scaling. Unlike SVM, `StandardScaler` is neither needed nor included.
- **Candidate Parameter Grid**:
  - The parameter grid is predeclared in `configs/phase1.toml` under `[training.random_forest]` before execution:
    - `n_estimators`: `[100, 200]`
    - `max_depth`: `[None, 10, 20]`
    - `min_samples_split`: `[2, 5]`
    - `class_weight`: `["none", "balanced"]`
  - 24 candidates per feature set × 5 feature sets = 120 candidate evaluations, each using five folds, giving 600 CV fits plus five final refits.
  - Identical candidate grids were evaluated across all 5 feature sets for a fair, controlled comparison.

## Feature Family Comparisons & Class Weighting Findings

### Unified Model Comparison Table

Reference baselines from Shourya's verified runs (`docs/training/svm-cv-summary.json`) are included alongside the 5 Random Forest feature studies:

| Model / Family | Descriptors | Dims | Mean CV Macro F1 | Fold Std | Selected Settings | Search/Refit Time (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Baseline: Dummy (Most Frequent) | COLOR+HOG+LBP | 1,806 | 0.0636 | 0.0002 | Most frequent class | 0.1 |
| Baseline: Linear SVM | COLOR+HOG+LBP | 1,806 | 0.6007 | 0.0284 | C=0.1, class_weight=None | 173.8 |
| Baseline: RBF SVM | COLOR+HOG+LBP | 1,806 | 0.6731 | 0.0339 | C=10.0, gamma=scale, class_weight=None | 143.8 |
| **Random Forest (HSV Color)** | COLOR | 32 | **0.7366** | 0.0260 | n_est=200, depth=20, min_split=2, weight=None | 13.4 |
| Random Forest (HOG) | HOG | 1,764 | 0.6252 | 0.0350 | n_est=200, depth=None, min_split=2, weight="balanced" | 81.0 |
| Random Forest (LBP) | LBP | 10 | 0.5069 | 0.0108 | n_est=100, depth=20, min_split=2, weight="balanced" | 10.6 |
| Random Forest (HSV+HOG) | COLOR+HOG | 1,796 | 0.7022 | 0.0389 | n_est=200, depth=20, min_split=2, weight="balanced" | 76.8 |
| Random Forest (Combined) | COLOR+HOG+LBP | 1,806 | 0.7090 | 0.0522 | n_est=200, depth=None, min_split=2, weight="balanced" | 77.7 |

### Observations & Hypotheses

1. **Observed HSV Color Performance**:
   - Random Forest trained on the 32-dimensional HSV color histogram obtained the highest mean CV macro F1 (**0.7366 ± 0.0260**), outperforming both RBF SVM (0.6731) and the higher-dimensional Random Forest models in this CV study.
   - Material categories in TrashNet exhibit distinctive chromatic distributions (e.g., cardboard brown, paper white/printed, glass green/clear). In 32 dimensions, orthogonal decision tree splits appear well-suited to separating these color clusters.
2. **Feature Subspace Dilution Hypothesis**:
   - Combining HOG and LBP with Color dropped the Random Forest score from 0.7366 to 0.7090, while increasing search/refit time from 13.4s to 77.7s.
   - *Hypothesis*: Because scikit-learn's Random Forest samples $m = \sqrt{p}$ features at each split ($\sqrt{1806} \approx 42$), the 32 color features constitute only ~1.8% of the candidate pool in the combined space. A plausible explanation is that individual tree nodes are often forced to choose among gradient and texture features even when color would be more discriminative. This is a working hypothesis consistent with the observed drop, though formal validation would require node-level feature importance and split-frequency inspection.
3. **Class Weighting Observations**:
   - Measured comparison of the best candidate within each feature set with `class_weight=None` vs. `class_weight="balanced"`:
     - HOG: 0.5739 (None) → 0.6252 (Balanced), an increase of +0.0513 in mean CV macro F1.
     - HSV+HOG: 0.6638 (None) → 0.7022 (Balanced), an increase of +0.0384.
     - Combined: 0.6683 (None) → 0.7090 (Balanced), an increase of +0.0407.
     - LBP: 0.5007 (None) → 0.5069 (Balanced), an increase of +0.0063.
     - HSV Color: 0.7366 (None) → 0.7340 (Balanced), a minor difference of -0.0027.
   - *Hypothesis on Minority-Class Sensitivity*: The outer training set contains moderate imbalance (e.g., 417 paper images vs. 96 trash images). Because macro F1 weights all 6 classes equally, unweighted trees in high-dimensional, less separable spaces may prioritize majority classes at the expense of minority-class recall. Balanced weighting penalizes errors inversely proportional to class frequencies, which appears beneficial where feature separability is lower, while having little impact when color separation is already strong. We report these as observed empirical differences across the CV folds rather than claiming formal statistical significance, which was not tested.

## Reproduction Commands

From the repository root with the activated environment:

```sh
# 1. Verify code formatting and linting
ruff check .

# 2. Run the complete test suite (72 unit tests)
python -m unittest discover -s tests -v

# 3. Reproduce Random Forest feature studies
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 waste-train rf --output artifacts/daksh-rf-runs --n-jobs 4 --estimator-n-jobs 1
```

## Handoff to Teammates

- **For Prajjwal Tripathi (Calibration & Evaluation)**:
  - Saved estimators are located in `artifacts/daksh-rf-runs/` and expect full 1,806-dim feature vectors.
  - The pipeline internally applies `ColumnTransformer` to select active features.
  - The saved estimators are uncalibrated (`calibrated=False`). Prajjwal will clone the selected pipeline architecture, perform group-aware calibration on outer training folds, choose the abstention threshold on outer validation data, and perform final frozen evaluation on the test split.
- **For Saksham Mighlani (Demo & Integration)**:
  - The trained `estimator.joblib` pipeline takes raw 1,806-dimensional feature arrays directly produced by `extract_features(image, config)` and handles column indexing automatically.

## Limitations & Viva Discussion Points

1. **Why Random Forest does not require StandardScaler**:
   - Decision trees evaluate split criteria using monotonic inequalities ($x_j \leq \theta$). Any strictly monotonic transformation $f(x)$ preserves the order of data points, leaving the Gini impurity / entropy calculation and chosen split thresholds functionally equivalent.
2. **Why HSV Color performs well on TrashNet but has deployment caveats**:
   - In laboratory / benchmark datasets, background illumination and object colors can be consistent. In practical real-world deployment, colored bins, lighting shifts, and transparent plastics against varying backgrounds can degrade color-only performance. HOG shape features provide geometric robustness even if raw benchmark CV favors color.
3. **The impact of feature subspace sampling ($m = \sqrt{p}$)**:
   - When concatenating 32 color features with 1,764 HOG features, the color features are heavily diluted during random candidate selection at each node. Feature weighting or hierarchical feature combinations represent potential improvements.
4. **Group-aware cross-validation necessity**:
   - Duplicate images (same physical object photographed from different angles) would cause severe data leakage if randomly distributed between train and test/validation folds. Stratified group K-fold ensures no duplicate group spans across train and validation splits.
5. **Macro F1 vs. Accuracy under Class Imbalance**:
   - Macro F1 averages the F1 scores across all 6 classes independently. If a classifier ignores the minority trash class, its macro F1 drops severely even if overall classification accuracy remains high.
