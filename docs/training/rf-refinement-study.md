# Random Forest Refinement Study

This study extends the original Random Forest baseline (HSV-only RF: mean 5-fold CV macro F1 **0.7366**; refinement HSV winner: **0.7419**) by evaluating Random Forest across the higher-dimensional representations defined in `src/waste_sorting/extended_features.py` (up to 2,760 dimensions).

The target of **0.86+ mean CV macro F1** was an exploratory aspiration, not a guarantee. Because these experiments reuse the frozen 5-fold cross-validation split on outer-training rows for model selection, all reported scores are exploratory training-CV selection metrics and should not be confused with final untouched-test generalization.

---

## 1. Experimental Rules & Frozen Split Protocol

- **Dataset Splitting**: Only outer-training rows (1,764 images across 6 classes) are used for model fitting, parameter selection, and diagnostic evaluation. Outer validation (380 images) and test (377 images) splits are never accessed or scored.
- **Group Isolation**: Exactly matches `docs/training/folds.json` (SHA-256: `6525c05675d2c9470659f2ad6a5721cadbe8558ac977ef7ef06dc858488b6625`). Duplicate image groups are strictly isolated within individual folds to prevent data leakage.
- **Extended Representation Provenance**:
  - Training matrix dimensions: $(1764, 2760)$.
  - Independent derived matrix SHA-256 fingerprint: `01b2deed507e7f5c808a3b249b29165ba41e59687b689b42c05cbe276756bbab`.
  - The shared baseline training fingerprint (`8b721ac54a69b6dfd39410ebec83caa9093ccad6747e8312e5e38dd1b1c572d5`) and the original cache remain completely unaltered.
- **Normalization Invariance**: Random Forest decision trees operate on monotonic axis-aligned rank splits; hence raw histogram counts/densities are passed directly without L2 normalization or feature scaling.

---

## 2. Parameter Sampling & Experimental Grid

To systematically test the **feature-subspace dilution hypothesis** (that high-dimensional descriptors dilute informative colour features when sampling random subsets at tree nodes), `max_features` was varied across `["sqrt", 0.25, 0.5, 1.0]`.

A reproducible candidate plan of **24 randomly sampled configurations** was generated using `ParameterSampler(random_state=42)` from:
- `n_estimators`: `[300, 600]`
- `max_depth`: `[None, 20, 40]`
- `min_samples_leaf`: `[1, 2, 4]`
- `max_features`: `["sqrt", 0.25, 0.5, 1.0]`
- `class_weight`: `[None, "balanced", "balanced_subsample"]`
- `min_samples_split`: fixed at `2`
- `bootstrap`: fixed at `True`

The identical 24 candidate configurations were evaluated across all 4 feature families:
1. `original_hsv`: 32 selected features.
2. `extended_global_color_texture`: 920 selected features.
3. `extended_global_color_spatial_texture`: 1,608 selected features.
4. `extended_spatial_color_texture`: 2,760 features (full vector).

**Execution Scope**: 4 families × 24 sampled candidates = **96 candidate evaluations**, each evaluated on 5 cross-validation folds = **480 CV fits**, plus 4 final full-training refits and 5 additional CV fits to generate the overall winner’s OOF diagnostics.

---

## 3. Results & Unified Model Comparison

### Feature Family Comparison

<!-- generated:families:start -->
| Family | Input dims | Selected dims | Mean CV macro F1 | Fold std | Pooled OOF accuracy | Family CV search wall time (s) | Best parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `original_hsv` | 1806 | 32 | 0.741915 | 0.017718 | Not recorded | 93.3 | `{"classifier__bootstrap": true, "classifier__class_weight": null, "classifier__max_depth": 20, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 300}` |
| `extended_global_color_texture` | 2760 | 920 | 0.831143 | 0.013520 | Not recorded | 1986.5 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |
| `extended_global_color_spatial_texture` | 2760 | 1608 | 0.826281 | 0.015241 | Not recorded | 3825.5 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": 0.25, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |
| `extended_spatial_color_texture` | 2760 | 2760 | 0.836955 | 0.024141 | 0.844671 | 5471.6 | `{"classifier__bootstrap": true, "classifier__class_weight": "balanced", "classifier__max_depth": null, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 600}` |

Family CV search wall time measures all 24 candidates × 5 folds (120 fits), including parallel scheduling, fitting, prediction and scoring. It excludes feature extraction, the final winning estimator refit, serialization and OOF diagnostics. Final-refit time was not recorded. The preserved raw `seconds` field has this meaning.
<!-- generated:families:end -->

Only the overall winning family has saved OOF diagnostics; unsupported accuracies for the other three families have been removed. The HSV pipeline accepts the full 1,806-dimensional vector and selects 32 columns internally.

*Winning fold scores*: `[0.8336, 0.8466, 0.8623, 0.8500, 0.7923]`

### Cross-Study Comparison

<!-- generated:comparison:start -->
| Model | Representation | Selected dims | Mean CV macro F1 | Fold std | Pooled OOF accuracy |
| --- | --- | --- | --- | --- | --- |
| Dummy baseline | HSV+HOG+LBP | 1806 | 0.063608 | 0.000195 | Not recorded |
| Original linear SVM | HSV+HOG+LBP | 1806 | 0.600692 | 0.028371 | Not recorded |
| Original RBF SVM | HSV+HOG+LBP | 1806 | 0.673063 | 0.033865 | Not recorded |
| Original HSV RF | HSV | 32 | 0.736622 | 0.025990 | 0.753968 |
| Compact RBF SVM | HSV+LBP | 42 | 0.804174 | 0.018910 | 0.819728 |
| Refined RF | Extended spatial color+texture | 2760 | 0.836955 | 0.024141 | 0.844671 |
| Extended RBF SVM | Extended global color+spatial texture | 1608 | 0.880263 | 0.021545 | 0.890023 |
<!-- generated:comparison:end -->

Accuracy sources and full-precision metrics are recorded in [the generated comparison](rf-refinement-results/cross-study-comparison.csv). Original baseline dummy/linear/RBF accuracies are not recorded and have been removed; F1 scores remain supported by their five-fold summaries.

---

## 4. Winning Model Diagnostics & Per-Class Performance

The winning Random Forest pipeline (`extended_spatial_color_texture`, 2,760 dims) achieved a **pooled OOF accuracy of 0.8447 (1,490 / 1,764 correct)** and a pooled macro F1 of **0.8377**.

### Per-Class Metrics (1,764 Training OOF Predictions)

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| cardboard | 0.9280 | 0.8688 | 0.8974 | 282 |
| glass | 0.8169 | 0.8075 | 0.8121 | 348 |
| metal | 0.7556 | 0.8217 | 0.7873 | 286 |
| paper | 0.8849 | 0.9423 | 0.9127 | 416 |
| plastic | 0.8349 | 0.7827 | 0.8080 | 336 |
| trash | 0.8506 | 0.7708 | 0.8087 | 96 |

### Out-of-Fold Confusion Matrix

```
True \ Pred      cardboard  glass  metal  paper  plastic  trash
cardboard              245      5     13     13        5      1
glass                    3    281     25      8       30      1
metal                    3     31    235      4       10      3
paper                    3      1     11    392        5      4
plastic                  5     21     22     21      263      4
trash                    5      5      5      5        2     74
```

### Key Diagnostic Observations
- **Paper & Cardboard**: Strongest recognition (F1: 0.9127 and 0.8974).
- **Metal vs. Glass Confusion**: 31 metal images misclassified as glass, and 25 glass images misclassified as metal, representing the single largest mutual confusion mode (likely due to specular reflections and metallic highlights).
- **Trash Class Performance**: In the initial 32-dim baseline, unweighted Random Forest had minority `trash` recall **0.5625** (54/96), recorded in `svm-study-results/summary.json` → `diagnostics.random_forest_color.per_class.trash.recall`. With the refined 2,760-dim representation and `class_weight='balanced'`, trash recall rose substantially to **0.7708** (74/96 correct), with an F1-score of **0.8087**.

---

## 5. Hypotheses & Technical Insights

1. **Subspace Dilution & Representation Richness**:
   - In the earlier 1,806-dim space, combining HOG with HSV and LBP yielded 0.7090 versus 0.7366 for HSV alone.
   - However, with the richer extended descriptors (288 colour bins across 6 regions + multi-scale LBP), Random Forest jumped by **+10.04 macro F1 points** (from 0.7366 to 0.8370).
   - In Families 2 and 3, `max_features=0.25` (sampling 25% of features per split) outperformed `max_features='sqrt'` (~1.8%), which is consistent with the feature-subspace hypothesis. Other hyperparameters vary across sampled configurations, so this is not an isolated causal test of `max_features`. In Family 4 (2,760 features), `sqrt(2760) = 52.5` provided sufficient feature diversity when combined with 600 estimators.
2. **Model Selection Recommendation for Phase 1**:
   - While the refined Random Forest model represents a major advancement over the original baseline (+10.04 macro F1 points, 0.8370 vs 0.7366), it remains below Shourya's Extended RBF SVM (**0.8803 macro F1**, 0.8900 accuracy).
   - **Recommendation**: For Prajjwal's Phase 1 calibration, abstention, and final test evaluation, the team should proceed with Shourya's **Extended RBF SVM (0.8803)** as the primary champion classifier. Daksh's refined Random Forest (0.8370) serves as a competitive non-linear ensemble benchmark and potential candidate for post-selection model ensembling.

---

## 6. Verification without fitting

<!-- generated:validation:start -->
All 96 candidates use the same 24 sampled configurations by index across four families. All 480 recorded fold scores reproduce their candidate means and population standard deviations (ddof=0; tolerance 1e-12; maximum error 9.71e-17). All family winners and the overall selection agree with the recorded tie-break rules.

Frozen fold SHA-256: `6525c05675d2c9470659f2ad6a5721cadbe8558ac977ef7ef06dc858488b6625`. Fold sizes: `[354, 354, 352, 352, 352]`. The 1,764 unique OOF rows match the training manifest, labels, duplicate groups and every frozen fold assignment; groups do not cross folds or outer splits.

Winner fold macro F1 reproduced from predictions: `[0.8336108303148674, 0.8466452726017942, 0.8622642553542916, 0.8500010472588052, 0.7922544828126225]`. Accuracy: **0.844671201814059** (1490 / 1,764); pooled macro F1: **0.8377107269285956**. Confusion counts and per-class metrics also agree.

Historical diagnostic accuracies were recomputed from integer confusion counts and checked against fold-size-weighted accuracies. The compact study's linear-SVM diagnostic belongs to its HSV+LBP model, not the original combined-feature linear baseline.

Recorded plan hash, shared runner folds, manifest identities/groups, and winner OOF assignments agree. Nonwinning candidate predictions/indices were not saved; their fold use is supported by the shared task construction, not a separate trace.

The completed experiment was not refitted. These are exploratory selection-CV scores, not untouched-test estimates. See `validation.json` for exact values and source hashes.
<!-- generated:validation:end -->

## 7. Reproduction

```sh
# Verify existing records and regenerate tables without fitting
python -m scripts.verify_rf_refinement --update-docs \
  --artifacts artifacts/daksh-rf-refinement

# Optional: a new full experiment (not required to verify these results)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m waste_sorting.rf_refinement \
  --output artifacts/new-rf-refinement-run --n-jobs 4 --seed 42

# Generate markdown summary and visual plots
python -m scripts.report_rf_refinement \
  --study artifacts/daksh-rf-refinement \
  --output docs/training/rf-refinement-results
```
