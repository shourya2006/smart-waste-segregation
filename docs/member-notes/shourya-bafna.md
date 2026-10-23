# Shourya Bafna: training protocol and SVM handoff

Prepared with the coding assistant for Shourya's assigned implementation.
Code, tests and experiments are assistant-assisted. Shourya approved committing
and pushing this baseline on 23 October 2026. The group remains responsible for
understanding the method.

## Implemented contribution

- `training.py`: strict cache/manifest alignment, training-only inputs, five
  saved stratified group-aware CV folds, shared search/result logging and CLI.
- `models/svm.py`: most-frequent dummy, linear SVM and RBF SVM pipelines with
  internal descriptor selection and fold-local SVM standardization.
- `tests/test_training.py`: identity/exclusion/integrity checks, outer and CV
  group isolation, frozen protocol validation, observed scaler fitting inputs,
  saved-pipeline loading and CLI reproduction.
- `docs/training/folds.json`: reproducible, reviewable training ID assignments.
- `docs/training/README.md`: teammate APIs and reproduction instructions.

The existing feature extractor and audited splits are reused. Random Forest,
feature-family comparisons, calibration, abstention, final evaluation and the
demo remain assigned to the corresponding teammates.

## Statistical protocol

The outer split contains 1,764 training images, 380 validation images and 377
test images. The loader verifies all included cache rows against the manifest,
then returns only the training rows. No model fit or model selection receives
the outer validation/test rows.

Training uses `StratifiedGroupKFold` with five folds, shuffling and seed 42.
Duplicate-group IDs keep related/uncertain image pairs together. Each fold
contains all six classes; stratification is approximate when groups constrain
the allocation. Training fingerprints and frozen image-ID assignments ensure
that SVM, Random Forest and feature studies use the same comparisons.

Six-class macro F1 is the unweighted mean of the six per-class F1 scores, with
undefined precision/recall contributions set to zero. The trash class therefore
receives the same metric weight as the much larger paper class. Each candidate
is evaluated on the same five training folds; selection uses their mean macro
F1. Report the population standard deviation across these five scores. It is
fold variation, not a confidence interval.

These are selection-CV results. Reusing them for hyperparameter choice introduces
selection optimism; they must not be labelled final-test performance. The best
pipeline is refitted on all outer training rows and saved uncalibrated.

## SVM theory and implementation choices

An SVM balances a large decision margin with penalties for classification
errors. The C parameter controls the relative penalty: smaller C applies
stronger regularization. LinearSVC here uses L2 regularization and squared-hinge
loss, learning one-vs-rest boundaries for six classes. Its finite iteration limit
is checked through convergence warnings in the default single-worker runner.

The RBF kernel is `K(x, z) = exp(-gamma * ||x-z||²)`. It permits nonlinear
boundaries; larger gamma makes similarity more local. SVC trains one-vs-one
binary subproblems, while its default multiclass decision output is arranged
one-vs-rest. Probability estimation is not enabled by this implementation.

HSV (32), HOG (1,764) and LBP (10) have different distributions and scales.
`StandardScaler` belongs inside the pipeline: on each CV fit it learns means
and standard deviations from that fold's training rows only, then transforms
the withheld rows. Fitting one scaler globally before CV would leak information.
After parameter selection, the final scaler is fitted on outer training only.

Class weighting compares no weighting with `balanced`; scikit-learn computes
balanced weights from the labels supplied to each fold's fit. The dummy model
predicts the most frequent class and provides a basic imbalance-aware reference
for interpreting macro F1. It does not need scaling.

The initial search is deliberately bounded and declared in configuration:
C in 0.1/1/10, class weight none/balanced, and RBF gamma scale/0.001/0.01.
Parameter search compares six linear and 18 RBF candidates plus one dummy.
No test performance was consulted to choose this grid.

## Reproduction and teammate handoff

Use the commands and common APIs in [the training guide](../training/README.md).
Daksh should use `load_training_data`, `load_or_create_folds`, `feature_columns`
and `run_experiment` for Random Forest and feature studies. The saved pipeline
expects full feature vectors and performs column selection internally.
Prajjwal must clone the complete chosen pipeline for group-aware calibration
on held-out training folds; the fitted uncalibrated artifact is not yet a
confidence-producing predictor for the demo.

Experiment metadata records source-content checksums, the base Git revision,
dirty state, dependency versions, settings, parameters, fold checksums, scores
and timing. This makes local pre-commit experiments distinguishable from runs
of a published commit. Raw data, feature caches and model binaries remain local.

## Limits and viva preparation

- Handcrafted features may confuse visually similar materials; performance on
  this dataset does not establish reliable disposal advice or unknown detection.
- Grouping reduces known duplicate leakage; the audit cannot prove that all
  related images have been detected.
- Feature-family selection belongs in training CV, using the same folds.
- Scaling, calibration and threshold selection each have separate fit scopes.
- Explain why macro F1, group-aware folds and a dummy baseline are necessary,
  why C/gamma affect SVM behavior, and why the test split stays outside selection.

## Observed results and validation

The verified real-data run on 23 October 2026 used all 1,806 features and the
same 1,764 training rows/1,729 groups throughout. Across five folds:

| Model | Mean macro F1 | Fold std | Best parameters |
| --- | ---: | ---: | --- |
| Dummy | 0.0636 | 0.0002 | Most frequent class |
| Linear SVM | 0.6007 | 0.0284 | C=0.1; no class weighting |
| RBF SVM | 0.6731 | 0.0339 | C=10; gamma=scale; no class weighting |

See [the training results](../training/README.md#verified-initial-results) for
full-precision summaries, all candidate scores and reproduction details.
These are parameter-selection CV results, not held-out performance estimates.
Random Forest and feature studies may change the final model choice.

Validation completed: `ruff check .`, all 68 unit tests, `python -m pip check`,
real CLI fold generation/reuse, full dummy/linear/RBF searches, saved-model
reloading, full candidate-score consistency, and both SVM scalers' sample counts
and means against outer-training rows. No validation/test predictions were made.
All source checksums in the recorded runs matched the completed implementation.
