# Daksh Batra — Random Forest and feature comparison

Follow shared-rules.md in this directory. Branch: `feat/random-forest`.
Review partner: Shourya Bafna. Depends on Aditya's features and Shourya's shared
training loader, fold assignments, and experiment schema.

## Implement

1. Add src/waste_sorting/models/random_forest.py with a scikit-learn-compatible
   Random Forest builder and bounded hyperparameter search. Explore tree count,
   depth, minimum leaf size, feature sampling, and unweighted/balanced class
   weighting within a documented compute budget.
2. Use the exact training rows, group-aware folds, seed, label order, and macro
   F1 criterion supplied by Shourya. Do not introduce your own split generator,
   fit learned transforms globally, or use validation/test scores for selection.
3. Add a feature-study runner for color-only, HOG-only, LBP-only, color+HOG,
   all-three, and removal of each descriptor family from the combined set.
   Reuse existing SVM builders for a fair comparison when they are available;
   do not copy or alter Shourya's SVM implementation to make results comparable.
4. Produce real per-fold results, means/variation, timing, parameters, feature
   dimensions, and a clear model/feature comparison table. Use the shared schema.
   State what the data actually supports; extra features need not improve results.
5. Add tests for integration with shared folds and caching, deterministic seeded
   fitting, requested class weights, experiment records, and subset selection.

## Acceptance and handoff

Run real Random Forest experiments and complete the training-only feature/model
comparison once dependencies exist. Keep models and caches local, and commit small
result summaries with exact commands. Write docs/member-notes/daksh-batra.md with
findings, Random Forest theory, bias/variance discussion, and limitations. Recommend
a candidate configuration using training CV evidence for Prajjwal; do not touch
the final test evaluation. Commit, push your branch, and open a PR.
