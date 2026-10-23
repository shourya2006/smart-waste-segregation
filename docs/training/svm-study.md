# SVM feature study and targeted refinement

This follow-up completes Shourya's feature-family comparison before selecting a
model for calibration. The existing all-feature SVM results and Daksh's Random
Forest results remain unchanged as reference baselines.

## Declared experiment plan

The plan below and `[svm_study]` in `configs/phase1.toml` were written before
running this follow-up. The same 1,764 training rows, 1,729 groups and five saved
folds are used throughout. Neither the feature cache nor fold assignments change.
No outer validation/test predictions are generated.

1. Screen linear and RBF SVMs on six feature sets: HSV (32), HOG (1,764), LBP (10),
   HSV+HOG (1,796), all three (1,806), and an additional compact HSV+LBP (42).
   Both use their original declared C/class-weight grids; RBF retains its original
   gamma grid. This is 144 candidate evaluations and 720 CV fits.
2. Select the top two RBF feature sets by mean training-fold macro F1. Ties are
   broken by feature-set name. Refine only these sets:
   - StandardScaler: C in 1/10/30/100, gamma in scale/0.25d⁻¹/4d⁻¹, class weight
     none/balanced, where d is selected feature dimension (24 candidates each).
   - Per-image square-root plus L2 normalization: the same C/weight grid, gamma
     in scale/1/10/50 (32 candidates each).
3. Recompute out-of-fold training predictions for the best linear SVM, best RBF
   SVM and the existing HSV Random Forest winner. Record per-class recall/F1,
   confusion matrices and accuracy, alongside mean-fold and pooled macro F1.
   These predictions are used for diagnostics, not unbiased final evaluation.

The bounded study stops after these grids: 256 candidate evaluations, 1,280 CV
fits, 16 outer-training refits and 15 diagnostic CV fits. It does not continue
adding experiments until a desired score appears. Selecting repeatedly on these
folds introduces optimism; a later frozen test evaluation is still required.

## Transform and convergence choices

All pipelines accept full 1,806-dimensional input and select columns internally.
StandardScaler remains inside CV. The root option takes the elementwise square
root of nonnegative descriptors, then L2-normalizes each image independently.
For pooled histograms this corresponds to a square-root/Hellinger-style
representation; it learns no statistics across images. Zero vectors remain zero;
negative/non-finite inputs are rejected. No extractor or image-processing change
is introduced.

`StrictLinearSVC` converts convergence warnings to errors inside each fit,
including parallel workers. The follow-up raises the linear iteration ceiling
to 100,000 without changing its objective, seed or stopping tolerance. It never
publishes a score from an unconverged linear fit. This ceiling is separate from
the original baseline's 20,000-iteration setting.

## Reproduce

```sh
python -m pip install --no-deps -e .
waste-data features
waste-train folds
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 waste-svm-study \
  --output artifacts/shourya-svm-feature-study --n-jobs 3
```

`--root`, `--cache`, `--config` and `--folds` accept alternative locations; relative
paths resolve against the repository root. Output directories must be new unless
`--resume` is supplied. Resume validates the complete saved plan, including source,
environment, configuration, folds and training fingerprint, before reusing runs.
Incomplete attempts are preserved under a distinct name before retrying.

The runner writes `study-plan.json` before any fitting. Each experiment retains
the shared runner's candidate CSV, fitted estimator and provenance summary.
`study-summary.json` and `candidate-results.csv` combine both stages. Diagnostics
contain only outer-training IDs and explicitly label score-selection optimism.
Model binaries and caches stay in ignored local directories; small verified
tables and metadata may be published after user approval.

## Completed original-feature study

All 256 candidates completed. The best RBF model uses HSV+LBP (42 selected
features), square-root/L2 normalization, C=30, gamma=10 and balanced weights:
mean five-fold macro F1 **0.804174**, fold std **0.018910**, pooled training-OOF
accuracy **0.819728**. The best linear model reaches 0.620295 macro F1.
The reference HSV Random Forest remains 0.736622; the original all-feature RBF
SVM remains 0.673063. These are comparable selection scores on the same folds.

Full results and diagnostic plots are in [svm-study-results](svm-study-results/README.md).
Every candidate's mean/std, all 16 saved pipelines, recorded source snapshot,
and all diagnostic image IDs/fold assignments were independently checked.
The best compact SVM still confuses metal/glass/plastic; trash recall is 0.6875.

The subsequent [extended-descriptor study](extended-svm-study.md) is a separate,
explicitly adaptive experiment motivated by these results and the requested
0.86 macro-F1 target. It does not retroactively change this study's plan or scores.
