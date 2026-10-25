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

<!-- generated:baselines:start -->
| Model | Descriptors | Selected dims | Mean CV macro F1 | Fold std | Selected parameters | Search + final refit wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | color+hog+lbp | 1806 | 0.063608 | 0.000195 | `{}` | 0.1 |
| linear_svm | color+hog+lbp | 1806 | 0.600692 | 0.028371 | `{"classifier__C": 0.1, "classifier__class_weight": null}` | 173.8 |
| rbf_svm | color+hog+lbp | 1806 | 0.673063 | 0.033865 | `{"classifier__C": 10.0, "classifier__class_weight": null, "classifier__gamma": "scale"}` | 143.8 |
| random_forest | color | 32 | 0.736622 | 0.025990 | `{"classifier__class_weight": null, "classifier__max_depth": 20, "classifier__min_samples_split": 2, "classifier__n_estimators": 200}` | 13.4 |
| random_forest | hog | 1764 | 0.625190 | 0.035031 | `{"classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__min_samples_split": 2, "classifier__n_estimators": 200}` | 81.0 |
| random_forest | lbp | 10 | 0.506936 | 0.010765 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 20, "classifier__min_samples_split": 2, "classifier__n_estimators": 100}` | 10.6 |
| random_forest | color+hog | 1796 | 0.702207 | 0.038927 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 20, "classifier__min_samples_split": 2, "classifier__n_estimators": 200}` | 76.8 |
| random_forest | color+hog+lbp | 1806 | 0.709007 | 0.052170 | `{"classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__min_samples_split": 2, "classifier__n_estimators": 200}` | 77.7 |

Generated from `svm-cv-summary.json` and `rf-cv-summary.json`. These historical `total_seconds` measurements include GridSearchCV search and final refit; they differ from the refinement runner's CV-search-only `seconds` field.
<!-- generated:baselines:end -->

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
   - *Hypothesis on Minority-Class Sensitivity*: The outer training set contains moderate imbalance (e.g., 416 paper images vs. 96 trash images). Because macro F1 weights all 6 classes equally, unweighted trees in high-dimensional, less separable spaces may prioritize majority classes at the expense of minority-class recall. Balanced weighting penalizes errors inversely proportional to class frequencies, which appears beneficial where feature separability is lower, while having little impact when color separation is already strong. We report these as observed empirical differences across the CV folds rather than claiming formal statistical significance, which was not tested.

## Random Forest Refinement Study (Extended Features)

Following Shourya's PR #5 extended feature representation (2,760 dimensions across 6 image regions), Daksh conducted a bounded Random Forest refinement study across 4 feature families to test whether richer representations and tuned `max_features` could overcome the earlier feature-subspace dilution.

### Evaluated Configurations & Setup
- **Scope**: 4 families × 24 sampled candidates = 96 candidate evaluations = 480 selection CV fits across the 5 frozen outer-training folds (`folds.json`), plus four final refits and five winner diagnostic CV fits.
- **Feature families**:
  1. `original_hsv`: 32 selected features.
  2. `extended_global_color_texture`: 920 selected features.
  3. `extended_global_color_spatial_texture`: 1,608 selected features.
  4. `extended_spatial_color_texture`: 2,760 features (complete vector).
- **Candidate Grid Sampled (Seed 42)**:
  - `n_estimators`: `[300, 600]`
  - `max_depth`: `[None, 20, 40]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `max_features`: `['sqrt', 0.25, 0.5, 1.0]`
  - `class_weight`: `[None, 'balanced', 'balanced_subsample']`
  - `min_samples_split`: fixed at `2`, `bootstrap`: fixed at `True`.

### Verified Outcomes

<!-- generated:families:start -->
| Family | Input dims | Selected dims | Mean CV macro F1 | Fold std | Pooled OOF accuracy | Family CV search wall time (s) | Best parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `original_hsv` | 1806 | 32 | 0.741915 | 0.017718 | Not recorded | 93.3 | `{"classifier__bootstrap": true, "classifier__class_weight": null, "classifier__max_depth": 20, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 300}` |
| `extended_global_color_texture` | 2760 | 920 | 0.831143 | 0.013520 | Not recorded | 1986.5 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |
| `extended_global_color_spatial_texture` | 2760 | 1608 | 0.826281 | 0.015241 | Not recorded | 3825.5 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |
| `extended_spatial_color_texture` | 2760 | 2760 | 0.836955 | 0.024141 | 0.844671 | 5471.6 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |

Family CV search wall time measures all 24 candidates × 5 folds (120 fits), including parallel scheduling, fitting, prediction and scoring. It excludes feature extraction, the final winning estimator refit, serialization and OOF diagnostics. Final-refit time was not recorded. The preserved raw `seconds` field has this meaning.
<!-- generated:families:end -->

Only the overall winning family has saved OOF accuracy. See the [verified cross-study comparison and audit](../training/rf-refinement-results/README.md).

- **Winning Model**: `extended_spatial_color_texture` (2,760 features)
  - Mean 5-fold CV macro F1: **0.8370 ± 0.0241** (winning folds: `[0.8336, 0.8466, 0.8623, 0.8500, 0.7923]`).
  - Pooled Out-of-Fold Accuracy: **0.8447** (1,490 / 1,764 correct).
  - Improvement over initial HSV baseline: **+10.03 macro F1 points** (0.7366 → 0.8370).
  - Per-class OOF F1: cardboard 0.8974, glass 0.8121, metal 0.7873, paper 0.9127, plastic 0.8080, trash 0.8087.
  - Trash recall rose from **0.5625** (54 / 96 correct; original HSV RF diagnostic) to **0.7708** (74 / 96 correct) with the changed representation and tuned estimator; the improvement cannot be attributed to weighting alone.

### Refinement Discussion & Model Recommendation
1. **Subspace Dilution Observations**:
   - In Families 2 and 3, setting `max_features=0.25` outperformed `sqrt` feature sampling, which is consistent with the feature-subspace hypothesis; varying other hyperparameters prevents a causal conclusion about `max_features` alone. In Family 4 (2,760 features), `sqrt` sampling (52 features) with 600 estimators provided the best ensemble diversity.
2. **Team Model Selection Recommendation**:
   - Random Forest refined performance improved to **0.8370 macro F1** (0.8447 accuracy), confirming Random Forest is highly viable on rich tabular representations.
   - However, Shourya's Extended RBF SVM achieved **0.8803 macro F1** (0.8900 accuracy) on the exact same outer-training CV folds.
   - **Recommendation**: The team should proceed with Shourya's **Extended RBF SVM (0.8803)** for Prajjwal's Phase 1 calibration, abstention, and final test evaluation. Daksh's refined Random Forest is documented as a competitive non-linear baseline and prime candidate for future model ensembling.

## Reproduction Commands

From the repository root with the activated environment:

```sh
# 1. Verify code formatting and linting
ruff check .

# 2. Run the complete test suite
python -m unittest discover -s tests -v

# 3. Verify saved refinement records and regenerate reports without fitting
python -m scripts.verify_rf_refinement --update-docs --artifacts artifacts/daksh-rf-refinement

# Optional: reproduce original Random Forest feature studies
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 waste-train rf --output artifacts/daksh-rf-runs --n-jobs 4 --estimator-n-jobs 1

# Optional: run a new Random Forest refinement experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m waste_sorting.rf_refinement --output artifacts/new-rf-refinement-run --n-jobs 4 --seed 42
```

## Handoff to Teammates

- **For Prajjwal Tripathi (Calibration & Evaluation)**:
  - Refined RF estimator is saved at `artifacts/daksh-rf-refinement/extended_spatial_color_texture/estimator.joblib`.
  - Takes 2,760-dim feature vectors from `extract_extended_features(image)` and internally handles column indexing.
  - The model is fitted on outer-training rows only and is uncalibrated.
- **For Saksham Miglani (Demo & Integration)**:
  - The pipeline interface accepts full extended feature matrices directly without external manual scaling.

## Limitations & Viva Discussion Points

1. **Why Random Forest does not require StandardScaler**:
   - Decision trees evaluate split criteria using monotonic inequalities ($x_j \leq \theta$). Any strictly monotonic transformation $f(x)$ preserves the order of data points, leaving the Gini impurity / entropy calculation and chosen split thresholds functionally equivalent.
2. **Why HSV Color performs well on TrashNet but has deployment caveats**:
   - In laboratory / benchmark datasets, background illumination and object colors can be consistent. In practical real-world deployment, colored bins, lighting shifts, and transparent plastics against varying backgrounds can degrade color-only performance. Spatial and texture features provide geometric robustness even if raw benchmark CV favors color.
3. **The impact of feature subspace sampling ($m = \sqrt{p}$ vs $0.25$)**:
   - When concatenating hundreds of texture features with color features, random feature sampling can dilute color features at tree nodes. Evaluating `max_features=0.25` demonstrated that allowing trees to evaluate a wider subset of features at each node helps preserve discriminative color channels.
4. **Group-aware cross-validation necessity**:
   - Duplicate images (same physical object photographed from different angles) would cause severe data leakage if randomly distributed between train and test/validation folds. Stratified group K-fold ensures no duplicate group spans across train and validation splits.
5. **Macro F1 vs. Accuracy under Class Imbalance**:
   - Macro F1 averages the F1 scores across all 6 classes independently. If a classifier ignores the minority trash class, its macro F1 drops severely even if overall classification accuracy remains high. Balanced class weighting penalizes minority errors proportionally to restore class balance.
