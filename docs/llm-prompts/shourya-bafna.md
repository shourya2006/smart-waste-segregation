# Shourya Bafna — shared training framework and SVM

Follow shared-rules.md in this directory. Branch: `feat/svm-baselines`.
Review partner: Daksh Batra. Depends on Aditya's feature interface/cache.

## Implement

1. Add src/waste_sorting/training.py for validated training-data loading,
   reproducible group-aware cross-validation, experiment records, and shared
   training utilities. Verify source/manifest/cache compatibility, preserve
   image-ID/label/group alignment, and exclude quarantined examples.
2. Create five seeded StratifiedGroupKFold folds using training rows only.
   Verify every fold has disjoint duplicate groups and all six classes where
   feasible. Persist fold membership or a verifiable digest so Daksh uses the
   exact same folds. Fail clearly if the requested protocol cannot be satisfied.
3. Add src/waste_sorting/models/svm.py with dummy, linear SVM, and RBF SVM
   baselines. Keep StandardScaler inside each SVM pipeline. Tune a bounded,
   documented range of C, RBF gamma, and class weights using macro F1 on the
   shared folds. Leave confidence calibration to Prajjwal.
4. Provide a reproducible training entry point. Log feature settings, per-fold
   macro F1, mean/variation, parameters, timing, seed, code revision, environment,
   and manifest checksum. Save model artifacts locally; track small real-run
   summaries and commands for reproduction. Establish this schema with Daksh.
5. Test training-only selection, group isolation, fold reproducibility, label
   and row alignment, cache mismatch rejection, and fold-local scaling behavior.

## Acceptance and handoff

Run the dummy and SVM baselines on real training features when upstream work is
available. Do not evaluate or tune on validation/test predictions, change the
split assignments, or select the final project model without comparing Daksh's
results. Write docs/member-notes/shourya-bafna.md with actual CV results, commands,
the shared protocol, SVM theory, and limitations. Commit your code/tests and real
summaries, push your branch, and open a PR. Do not implement Random Forest or
Prajjwal's calibration module.
