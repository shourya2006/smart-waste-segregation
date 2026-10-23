# Shared instructions for every member's coding assistant

Repository: https://github.com/shourya2006/smart-waste-segregation

Read this file, the member-specific brief, README.md, CONTRIBUTING.md,
docs/phase-1-plan.md, and docs/team-assignments.md before implementing.
Follow applicable repository instructions and inspect the current code rather
than assuming the interfaces below already exist.

## Scope and existing work

Implement the assigned Phase 1 component, not the whole project. Phase 1 is
classical ML using handcrafted image descriptors. Do not add neural networks,
pretrained CNN features, or Phase 2 work.

Data acquisition, audit, duplicate review, fixed splits, and training-only EDA
already exist. There are 2,521 included images: 1,764 training, 380 validation,
and 377 test. The original archive contains 2,527 images; six label-conflicting
copies are excluded. Preserve docs/dataset/manifest.csv, its assignments and
checksums, the exclusion decisions, and the raw source images. Verify against
the current tracked summaries rather than hardcoding the counts into loaders.
If a real integrity defect appears, document it and notify the team before
changing the evaluation protocol. Do not redo completed work for commit counts.

## Branches and identity

Clone the repository if needed; otherwise inspect the existing checkout and
preserve any local changes. Fetch origin and start the assigned branch from the
latest origin/main, or safely continue that member's existing task branch.
Do not reset an existing checkout or overwrite teammates' work.

Use the human member's own configured Git identity and authenticated account.
Verify author configuration before committing. Never invent an email, impersonate
a teammate, change global Git identity, or rewrite existing authorship. If identity
or authentication is missing, finish useful local work and request the missing
setup rather than manufacturing attribution.

Make meaningful commits, push only the assigned branch, and open a pull request
targeting main. Do not force-push, push directly to main, or merge the pull request.
Request the designated review partner if their GitHub handle is known; otherwise
name the partner in the PR description without guessing a handle. If the current
assistant provides a PR attachment tool, attach the created PR using that tool.

## Setup, checks, and data

Use Python 3.12 and the environment instructions in README.md. Install the locked
dependencies before the editable package. Use `waste-data download` to obtain
the checksum-pinned dataset when images are needed. Audit commands can verify
the source; do not replace committed review decisions or regenerate different
split assignments. Run `ruff check .`, the existing unittest suite, and relevant
new behavior tests. Keep existing checks passing.

Never commit raw images, credentials, feature-cache binaries, or trained model
binaries. Store generated caches/models under the ignored data/processed/ or
artifacts/ directories. Track small experiment summaries and reproduction commands.
Coordinate changes to pyproject.toml, requirements.lock, configs/phase1.toml,
and shared entry points; avoid unrelated dependency upgrades.

## Common interfaces and evaluation rules

- The `config` argument is the complete mapping loaded from configs/phase1.toml.
  Document additional options without silently changing existing defaults.
- Om owns `preprocess_image(image, config)`: input is a PIL image, output is an
  EXIF-corrected RGB PIL image letterboxed to the configured width and height.
- Aditya owns `extract_features(image, config)`: use Om's preprocessing and return
  a finite, deterministic, one-dimensional float32 feature vector. Feature order
  is HSV, HOG, then LBP, selecting subsets without changing relative order.
- Shourya owns the training loader, group-aware fold assignments, and experiment
  record schema. Both model implementations use the same training examples,
  folds, class order, seed, features, and selection metric.
- Fit StandardScaler and other learned transforms inside each training fold.
  Deterministic descriptors can be cached, but do not globally fit transformations.
- Select feature/model settings by training cross-validation macro F1. Carry
  duplicate-group IDs through every fold and every comparison. Keep validation
  data for abstention selection and test data for final evaluation.
- Prajjwal owns calibrated saved-model metadata and `predict_image(image, artifact)`.
  Prediction returns class probabilities in a documented order, predicted class,
  maximum probability, and an abstained flag. Feature configuration, manifest
  checksum, class order, threshold, environment, and code revision accompany models.
- Saksham calls this prediction function in the demo; never implement a second
  preprocessing/descriptor path or display random scores as model output.

## Dependencies and completion

Work can start on interfaces, fixtures, tests, and documentation before upstream
components are merged. Do not create competing implementations of another member's
component. If an upstream PR exists, inspect it and document the dependency; use
an explicitly documented dependent branch if needed. Otherwise finish independent
parts and clearly identify what integration remains. Do not leave mocks masquerading
as a completed working pipeline.

Write a technical handoff under docs/member-notes/ using the member's name.
Include implemented files, commands, tests, actual results, theory for the viva,
report-ready methodology, limitations, and dependencies. Cite actual literature
or official technical documentation when used; never invent references or scores.
Add a contribution-ledger row describing the member's actual completed work and
link the resulting commit or PR. Do not invent contributions for anyone else.

Finish by reporting: changes, verification, actual experiment results if any,
remaining dependencies, commit hashes, branch, and PR URL. A brief and tests alone
do not count as completion of a component requiring real-data integration.
