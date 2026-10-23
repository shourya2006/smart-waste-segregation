# Six-person Phase 1 work division

The data-preparation milestone is already implemented. Divide the remaining
work below; do not recreate completed work just to generate commits.
Names were supplied by Shourya Bafna. GitHub usernames remain to be recorded
for collaborators other than Shourya (`shourya2006`).
The source course brief requires exactly five students; this requested six-person
division is not evidence of approval for a six-person submission.

## Ownership and concrete deliverables

| Member | Owns | Deliverables | Suggested branch |
| --- | --- | --- | --- |
| Om Yadav | Data verification and preprocessing | Independently review uncertain duplicate pairs; verify manifest and source hashes; implement EXIF correction, RGB conversion, aspect-preserving 128×128 letterboxing; test orientation, padding, and corrupt-input behavior | `feat/preprocessing` |
| Aditya Bharadwaj | Handcrafted features | Implement normalized HSV histograms, HOG, uniform LBP, configurable combinations, and a cache keyed by manifest/configuration; document dimensions and descriptor examples | `feat/image-features` |
| Shourya Bafna | Training protocol and SVM | Implement shared training-input validation and group-aware folds, dummy baseline, linear/RBF SVM with fold-local scaling, parameter search, and reproducible experiment logging | `feat/svm-baselines` |
| Daksh Batra | Random Forest and feature studies | Implement Random Forest and parameter search using the shared folds; compare feature sets and class weighting with the same selection metric; produce a comparison table with fold variation | `feat/random-forest` |
| Prajjwal Tripathi | Confidence and evaluation | Calibrate the selected full pipeline on held-out training folds; select abstention on validation data; implement per-class metrics, reliability plots, coverage analysis, final test evaluation, and failure review | `feat/model-evaluation` |
| Saksham Mighlani | Demo and submission integration | Implement Gradio upload UI using the shared prediction pipeline, uncertainty response and limitations note; test saved-model integration; assemble shared report sections, slides, and reproduction guide | `feat/demo-integration` |

Everyone writes the report section and slides for their own technical work.
Saksham Mighlani assembles these contributions rather than writing everyone's content.
Every member must contribute substantive implementation commits and understand
the full system for the individual viva.

## Proposed file ownership

These are agreed implementation targets, not files already implemented.
Keep experiments out of the existing audited split files unless a documented
data correction is necessary before model evaluation begins.

| Member | Primary new files or folders |
| --- | --- |
| Om Yadav | `src/waste_sorting/preprocessing.py`, `tests/test_preprocessing.py` |
| Aditya Bharadwaj | `src/waste_sorting/features.py`, `tests/test_features.py` |
| Shourya Bafna | `src/waste_sorting/training.py`, `src/waste_sorting/models/svm.py`, associated tests |
| Daksh Batra | `src/waste_sorting/models/random_forest.py`, feature-study runner and associated tests |
| Prajjwal Tripathi | `src/waste_sorting/evaluation.py`, `src/waste_sorting/prediction.py`, associated tests |
| Saksham Mighlani | `app.py`, demo integration tests, report assembly and presentation sources |

Shourya Bafna coordinates shared training entry points. Aditya Bharadwaj coordinates changes
to descriptor configuration. Discuss shared configuration and dependency edits
in the pull request so two members do not silently overwrite each other's choices.

## Interfaces to agree before coding

- Om Yadav provides `preprocess_image(image, config)` returning one RGB PIL image
  at the configured size. The same function is used for features and demo input.
- Aditya Bharadwaj provides `extract_features(image, config)` returning a fixed-length
  finite NumPy vector with documented ordering. Features remain unscaled; scaling
  belongs inside the model pipeline. Cache metadata includes image IDs, hashes,
  feature settings, and feature order.
- Shourya Bafna provides a shared training loader and fixed group-aware fold indices.
  The loader carries labels, image IDs, and duplicate groups together and excludes
  quarantined rows. Both model owners use the same folds and experiment schema.
- Shourya Bafna and Daksh Batra return scikit-learn-compatible pipelines and record feature
  settings, hyperparameters, fold macro F1 scores, timing, seed, code revision,
  and manifest checksum. Neither selects models using test predictions.
- Prajjwal Tripathi owns saved-model metadata and `predict_image(image, artifact)`.
  Its result includes class probabilities, predicted class, maximum probability,
  and an abstained flag. Calibration includes the complete scaling/model pipeline.
- Saksham Mighlani calls that prediction function rather than reimplementing preprocessing
  or descriptors in the app. Clearly marked fixtures can support early UI work;
  actual demo results must come from a trained model.

## Work order and handoffs

1. Everyone clones and runs the current checks. Om Yadav independently verifies
   the prepared data; Aditya Bharadwaj agrees preprocessing/feature interfaces with them.
   Shourya Bafna and Daksh Batra agree the experiment schema. Prajjwal Tripathi defines evaluation
   functions and tests. Saksham Mighlani prepares integration and report structure.
2. Merge preprocessing, then features. Shourya Bafna and Daksh Batra run model experiments
   against the same training fold assignments. Synthetic fixtures may support
   code development before the real feature cache exists.
3. Choose the feature/model settings by training cross-validation. Prajjwal Tripathi
   calibrates with training-only held-out folds and selects the abstention
   threshold on validation data. Freeze the protocol before final test evaluation.
4. Saksham Mighlani integrates the saved model. Complete final evaluation, error analysis,
   report sections, slides, reproduction, and viva practice together.

## Meaningful commit examples

These are examples, not a required commit quota. Make a commit for a coherent,
working improvement and keep related tests with its implementation when practical.

| Member | Example commit subjects |
| --- | --- |
| Om Yadav | `Add shared RGB letterbox preprocessing`; `Verify duplicate decisions and source integrity` |
| Aditya Bharadwaj | `Implement HSV HOG and LBP descriptors with dimension checks`; `Cache feature combinations with source metadata` |
| Shourya Bafna | `Add group-aware training folds and dummy baseline`; `Train and tune SVM pipelines with fold-local scaling` |
| Daksh Batra | `Implement Random Forest training with shared folds`; `Compare feature families and balanced class weights` |
| Prajjwal Tripathi | `Add calibration and validation abstention analysis`; `Evaluate frozen pipeline with class-wise errors` |
| Saksham Mighlani | `Integrate image upload with saved-model predictions`; `Assemble shared report and reproduction guide` |

Do not copy unchanged files, reattribute existing commits, or split trivial edits
to manufacture activity. No one should commit using another member's identity.

## Review partners and completion criteria

Review pairings: Om Yadav ↔ Aditya Bharadwaj, Shourya Bafna ↔ Daksh Batra, Prajjwal Tripathi ↔ Saksham Mighlani.
Each member reviews the partner's pull request and runs a relevant reproduction
check. Review comments complement implementation commits; each member still owns
their own deliverables.

A member's task is complete when their implementation works with the shared
interfaces, relevant tests pass, assumptions are documented, their report section
is written, and the contribution ledger links to their actual commits or pull
requests. Preserve test-set isolation and do not upload raw data or model binaries.
