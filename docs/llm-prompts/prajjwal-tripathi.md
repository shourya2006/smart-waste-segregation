# Prajjwal Tripathi — confidence, evaluation, and prediction

Follow shared-rules.md in this directory. Branch: `feat/model-evaluation`.
Review partner: Saksham Mighlani. Depends on the shared training protocol and a
candidate feature/model configuration selected by training CV.

## Implement

1. Add src/waste_sorting/evaluation.py for class-wise precision/recall/F1,
   macro F1, accuracy, confusion matrices, probabilistic log loss, reliability
   diagrams, abstention coverage/accepted accuracy, and qualitative error analysis.
   Test metrics with known expected results, missing predicted classes, and the
   zero-accepted-predictions case. State the multiclass calibration plotting method.
2. Calibrate the chosen full pipeline using held-out training folds and the
   shared duplicate-group-aware CV indices. Keep scaling/learned preprocessing
   inside the fitted pipeline. Use sigmoid calibration as the initial method.
   Do not calibrate on in-sample scores or use test data to fit calibration.
3. On validation data, compare confidence thresholds, coverage, accepted accuracy,
   and class-wise acceptance/recall. Document the threshold-selection criterion
   before applying it, including what happens if the desired tradeoff is not met.
4. Define a saved artifact containing the calibrated pipeline, descriptor and
   preprocessing configuration, label order, threshold, seed, manifest checksum,
   environment, and code revision. Keep binaries local and document reproducible
   artifact creation for Saksham.
5. Add src/waste_sorting/prediction.py with `predict_image(image, artifact)`.
   Reuse Om/Aditya's exact image pipeline. Return documented class probabilities,
   predicted class, maximum probability, and an abstained flag. Test persistence
   round-tripping and training/inference consistency.
6. Provide an explicit final-evaluation command. Only run it after the feature/model
   settings, calibration, threshold, and protocol are fixed. Record the exact
   artifact identity. Report full six-class test metrics before abstention and
   selective results separately. Do not repeatedly optimize from test results.

## Acceptance and handoff

Implement and test independent evaluation utilities immediately. If trained
dependencies are missing, do not fabricate a completed calibrated artifact or
final scores; identify the dependency. Once available, run the complete valid
procedure and record real results in docs/member-notes/prajjwal-tripathi.md,
including confidence limitations, failure cases, and viva theory. Commit code,
tests, and small actual-result summaries, push your branch, and open a PR.
