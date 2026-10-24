# Contributions

The six members below were supplied by the user. Verified accounts are Shourya
Bafna (`shourya2006`) and Om Yadav (`yadavom7345`). Other GitHub usernames are
pending. Each member must configure their own Git author details. The course team-size requirement is documented in
[the implementation assignments](team-assignments.md).

Ownership does not replace the individual understanding required for the viva.
Detailed deliverables and dependencies are in [team-assignments.md](team-assignments.md).

| Owner | Primary responsibility | Evidence |
| --- | --- | --- |
| Om Yadav | Independent data verification and image preprocessing | Data review, preprocessing code, tests |
| Aditya Bharadwaj | HSV, HOG, LBP, and feature caching | Descriptor code, dimension checks, visualizations |
| Shourya Bafna | Shared training protocol, dummy baseline, SVM | Group-aware folds, SVM runs, theory |
| Daksh Batra | Random Forest and feature comparisons | RF runs, feature experiments, comparison tables |
| Prajjwal Tripathi | Calibration, abstention, and final evaluation | Metrics, reliability plots, failure analysis |
| Saksham Mighlani | Photo upload demo and submission integration | Demo code, integration checks, report assembly |

## Existing repository history

The initial implementation was prepared with the Codex assistant under the
repository owner's configured Git identity. This work predates the six-person
assignment and must not be retrospectively claimed as another student's work.

| Commit | Work already completed |
| --- | --- |
| `1b4b6d2` | Repository foundation and Phase 1 roadmap |
| `eba1fdb` | TrashNet acquisition, audit, duplicate review, fixed splits, EDA, 19 tests |

## Individual work ledger

Add a row only after the work exists. Use each member's real name and link to
their own commit or pull request; do not count this assignment as completed work.

| Contributor | Date | Implemented or reviewed work | Commit or pull request |
| --- | --- | --- | --- |
| Om Yadav | 2026-10-23 | Assistant-assisted RGB letterbox preprocessing, 13 behavior tests, and in-memory reproduction of the approved data protocol | [Preprocessing branch](https://github.com/shourya2006/smart-waste-segregation/tree/feat/preprocessing) |
| Aditya Bharadwaj | 2026-10-23 | Assistant-assisted HSV/HOG/LBP feature extraction (1,806 dimensions), reproducible caching with stale cache detection, 16 unit tests, and validation on 2,521 TrashNet images | [Draft PR #2](https://github.com/shourya2006/smart-waste-segregation/pull/2) |
| Shourya Bafna | 2026-10-23 | Assistant-assisted training-input validation, five shared group-aware folds, dummy/linear/RBF SVM searches, fold-local scaling tests and teammate handoff | [SVM baseline branch](https://github.com/shourya2006/smart-waste-segregation/tree/codex/svm-baselines) |
| Daksh Batra | 2026-10-23 | Assistant-assisted Random Forest pipeline, 5-family feature comparison study (120 evaluated candidates), class weighting experiments, 4 unit/integration tests, and documented handoff | [Random Forest branch](https://github.com/shourya2006/smart-waste-segregation/tree/codex/random-forest) |
| Prajjwal Tripathi | 2026-10-24 | Group-aware Platt calibration, validation abstention threshold sweep and selection, frozen test set evaluation, prediction interface, 18 unit tests, and reproducible report artifacts | [Calibration & evaluation branch](https://github.com/shourya2006/smart-waste-segregation/tree/feat/calibration-evaluation) |

Record completed work with the contributor's name, a short description, date,
and commit or pull-request link. Each member should review another component.
All report sections should identify the actual contributors.

## SVM follow-up

Shourya Bafna, with the coding assistant: SVM feature/normalization study (256 candidates), opt-in extended colour/texture SVM study (72 candidates), training-only diagnostics and reproducible reports; 0.8803 mean CV macro F1 and 82 passing tests. Shourya approved committing, pushing and opening the follow-up PR. Implementation and experiment records: [SVM feature study branch](https://github.com/shourya2006/smart-waste-segregation/tree/codex/svm-feature-study).

## Random Forest refinement

Daksh Batra, with the coding assistant: Random Forest refinement study evaluating 4 feature families (original HSV, extended global color+texture, extended global color+spatial texture, extended spatial color+texture) across 24 predeclared parameter configurations sampled deterministically with seed 42 (96 candidate evaluations, 480 selection CV fits, 4 final refits, 5 winner diagnostic CV fits). Mean CV macro F1 reached **0.8370 ± 0.0241** (pooled OOF accuracy **0.8447**), representing a +10.03 point gain over the initial HSV baseline. Recorded results were audited without refitting: all 96 candidate means/stds and winner OOF scores reproduce; historical accuracies and timing labels are corrected. Generated comparisons, source hashes, and verification results are recorded in [Random Forest refinement documentation](training/rf-refinement-study.md). Implementation branch: `codex/rf-refinement`.

## Confidence calibration, abstention and evaluation

Prajjwal Tripathi: Implementation of group-aware sigmoid calibration (`CalibratedClassifierCV`, 5 frozen training folds, `ensemble=True`) on the selected extended RBF SVM (`global_color_spatial_texture`, C=10, gamma=10). Validation-only abstention threshold sweep and selection (threshold 0.52 achieving 98.16% coverage and 90.08% accepted accuracy). Frozen evaluation on untouched test data (377 images) producing **0.8999 macro F1** (0.9019 accuracy, 0.4599 log loss, 0.1828 multiclass Brier score, 0.1325 ECE) unconditional, and **0.9129 macro F1** (98.41% coverage, 91.37% accuracy) post-abstention. Implemented shared `predict_image(image, artifact)` interface, comprehensive unit tests (18 tests, 112 passing in total suite), final-evaluation overwrite guard, and generated reproducible tables and visualization plots. Branch: `feat/calibration-evaluation`.
