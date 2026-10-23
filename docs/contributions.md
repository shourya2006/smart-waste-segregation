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

Record completed work with the contributor's name, a short description, date,
and commit or pull-request link. Each member should review another component.
All report sections should identify the actual contributors.
