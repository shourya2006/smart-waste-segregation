# Phase 1 implementation roadmap

## Assignment requirements

The supplied course instructions specify ML-only evaluation in Phase 1, teams
of exactly five, an individual viva, a GitHub repository, a presentation, and a
LaTeX/IEEE report. Evaluation includes literature review, dataset quality/EDA,
feature engineering, theoretical rigor, model application, and code quality.

Project 12 specifies color histograms, HOG, and LBP with SVM and Random Forest;
duplicate checks before stratified splitting; macro F1, class recall, confidence
calibration, qualitative failures, and low-confidence abstention in the demo.
TrashNet is the recommended starter dataset. The transfer learning comparison
belongs to Phase 2 under the course's evaluation structure.

## Recommended implementation milestones

The requested six-person implementation assignments are maintained in
[team-assignments.md](team-assignments.md). The course's five-person requirement
above remains a separate constraint to resolve with faculty.

The split ratios, feature settings, and sequence below are implementation
choices. They are not additional faculty requirements.

- [x] Initialize the repository, package metadata, configuration, and workflow.
- [x] Validate and lock the working Python environment.
- [x] Record dataset provenance and verify all image files.
- [x] Identify exact duplicates; review perceptual-hash candidates and document limits.
- [x] Create an audited manifest and preserve duplicate decisions.
- [x] Create approximately 70/15/15 stratified splits, respecting detected groups.
- [x] Explore training class balance, samples, dimensions, brightness, and ambiguity.
- [x] Implement consistent orientation correction, RGB conversion, and letterboxing.
- [x] Extract HSV histograms, HOG, and uniform LBP with documented dimensions.
- [x] Implement shared training-input validation and five frozen group-aware CV folds.
- [x] Implement dummy, linear SVM and RBF SVM training with fold-local scaling and logging.
- [x] Complete Random Forest training and compare the initial model families.
- [x] Compare five Random Forest feature families and tune using training-only cross-validation.
- [x] Complete SVM feature/normalization search and extended-descriptor study (0.8803 selection-CV macro F1).
- [x] Calibrate probabilities using held-out training folds.
- [x] Select an abstention threshold using validation data and inspect coverage.
- [x] Freeze and evaluate on the untouched test set.
- [x] Report macro F1, per-class metrics, confusion matrix, and calibration.
- [ ] Analyze mistakes and optionally evaluate independent everyday photos (mistake analysis completed from saved test predictions; external everyday photos deferred to Phase 2).
- [x] Integrate a Gradio photo upload prototype with an uncertainty response and limitations note (real-model integration verified using the preserved artifact and generated images; see demo-verification.md).
- [x] Complete literature review, LaTeX/IEEE report sources, slides, and viva preparation (report PDF compiled and visually verified; editable presentation and presenter notes prepared).
- [ ] Have a teammate reproduce the final results from a clean checkout (reproduction guide documented; teammate execution pending).

## Experiment protocol

All candidate models use the same audited split assignments. Related images
stay together in both the outer split and cross-validation folds. Scaling, PCA
if introduced, and any learned preprocessing are fitted only within training
folds. Optional augmentations are generated only for training-fold images.

Compare color-only, HOG-only, LBP-only, color+HOG, and all-three feature sets.
Select settings by mean cross-validation macro F1. Record fold variation,
parameters, runtime, feature dimensions, source revision, and environment.

Calibration must use predictions on examples withheld from the corresponding
base-model fit. Keep the complete scaling/classification pipeline inside this
procedure. Select the abstention threshold on validation data and freeze it
before the test evaluation. Report six-class performance before abstention and
coverage/accepted accuracy separately. Do not tune after viewing test results.

## Suggested four-week schedule

| Week | Milestone |
| --- | --- |
| 1 | Environment, literature, dataset audit, duplicate review, splits, EDA |
| 2 | Preprocessing, feature extraction, initial model comparison |
| 3 | Feature experiments, tuning, calibration, final evaluation |
| 4 | Demo, report, slides, reproduction, individual viva preparation |

No evaluation date is specified in the supplied instructions. Confirm dates
through the course's designated communication channel.
