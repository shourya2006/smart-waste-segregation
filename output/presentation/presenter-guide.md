# Presentation guide

Suggested main presentation: about 13 minutes. Slides 16–18 are backup material.
Use the embedded speaker notes in PowerPoint Presenter View.

## 1. Smart Waste Segregation

**Shourya Bafna** — 20 seconds

Introduce the educational material classifier. State that this is the classical machine learning phase. The measured result belongs to a controlled TrashNet test split, so it is not a universal disposal recommendation. Hand over to Om for the problem and data.

## 2. A material suggestion from a photograph

**Om Yadav** — 30 seconds

Explain the six labels. Local rules, contamination and composite materials determine disposal decisions, so we present material suggestions only. The baseline deliberately uses manual descriptors and classical SVM/Random Forest models. Transfer learning belongs to Phase 2.

## 3. 2,521 usable images after the audit

**Om Yadav** — 50 seconds

Start with 2,527 readable images. Three conflicting-label duplicate groups contain six files, all excluded without relabeling. The remaining counts shown here come from the manifest. Paper is the largest category and trash the smallest. Training imbalance is 416 versus 96, which motivates macro F1.

## 4. Training, threshold selection and test stay separate

**Om Yadav** — 40 seconds

The outer split is fixed and group-aware. Training supports all model searches and calibration. Validation selects only the abstention threshold. Test measures the frozen pipeline. We preserve detected same-item and uncertain duplicate groups. Perceptual hashing can miss repeated objects, so do not claim perfect object-identity isolation. Hand to Aditya.

## 5. Baseline features capture color, shape and texture

**Aditya Bharadwaj** — 50 seconds

Explain each handcrafted descriptor using intuition. HSV summarizes color. HOG summarizes local edge directions, with unsigned orientations and normalized overlapping blocks. LBP compares each pixel with nearby intensities. The baseline concatenates 32, 1764 and 10 values. The common preprocessing preserves aspect ratio and white-pads a 128 by 128 canvas. Caches validate manifest and extractor identity.

## 6. Spatial texture adds detail to the representation

**Aditya Bharadwaj** — 55 seconds

The extended extractor uses a 256-square canvas and six regions: full image, fixed foreground mask and four quadrants. Each region has 288 color and 172 texture values. This produces 2760 raw inputs. The winner chooses color from full image plus foreground and texture from all regions: 576 plus 1032 equals 1608. It does not include HOG. The mask is a heuristic and can miss pale objects. Hand to Shourya.

## 7. The extended SVM reaches 0.8803 mean CV F1

**Shourya Bafna** — 65 seconds

The stages share the same five training folds. The baseline RBF SVM scored 0.6731. A compact HSV plus LBP representation reached 0.8042. The extended selected subset reached 0.8803. These are selection scores, not test scores. The selected pipeline applies square root then per-image L2 normalization and an RBF kernel with C=10 and gamma=10. Its five-fold standard deviation is 0.0215. The extended search evaluated 72 candidates after the 256-candidate compact follow-up. Repeated fold reuse can produce optimism, which motivates the frozen test.

## 8. Random Forest improves with richer descriptors

**Daksh Batra** — 55 seconds

Describe the RF experiment and the value of a second model family. All four refined families use the same 24 sampled configurations and frozen folds. The best refined RF uses all 2760 spatial color and texture features and reaches 0.8370. Its selected settings are 600 trees, unlimited depth, balanced class weights, bootstrap, and square-root feature sampling. Do not claim max_features=0.25 for the winning family.

## 9. More features do not always improve the model

**Daksh Batra** — 40 seconds

The initial HSV-only forest outperformed the combined baseline despite having far fewer dimensions. The extended SVM also favored a subset over all extended features. This supports feature-family comparisons. Feature dilution is a possible mechanism, not proven causation. The SVM has higher mean selection F1 than refined RF, but we did not run a formal significance test. Hand to Prajjwal for confidence and evaluation.

## 10. Calibration and the abstention rule

**Prajjwal Tripathi** — 60 seconds

The selected SVM hyperparameters stay frozen. For each of five training folds, a base model learns from four folds and a sigmoid calibrator learns from the withheld fold. We average calibrated probabilities. This is training-only calibration, not validation or test fitting. On 380 validation images we choose a threshold from 101 candidates. At 0.52, 373 are accepted and 336 are correct, satisfying 90% accepted accuracy and 80% coverage targets. The same folds selected hyperparameters earlier, so training calibration diagnostics are not independent.

## 11. Frozen test result: 0.8999 macro F1

**Prajjwal Tripathi** — 45 seconds

Clearly distinguish unconditional performance on all 377 images from selective performance on 371 accepted images. Accuracy rises from 90.19 to 91.37 percent on the accepted subset, a 1.19 percentage-point difference. Six predictions abstain: five errors and one correct prediction. These figures describe the controlled TrashNet test partition. They are not a promise for everyday photos.

## 12. Errors concentrate in visually similar materials

**Prajjwal Tripathi** — 35 seconds

Read the matrix by row: true class on rows and predictions on columns. Metal-to-glass is the largest individual off-diagonal with six examples. Plastic has the lowest recall, 61/72, with errors across several categories. Trash recall is high but based on only 20 test images. This analysis uses saved predictions and does not tune the model. Hand to Saksham for the demo.

## 13. Photo upload demo

**Saksham Mighlani** — 60 seconds

Switch to the live local application if available. Upload one previously prepared photo, show all six probabilities, and explain the fixed threshold. Clear and replace the input to show stale-state removal. Do not change model settings or treat the demonstration as an accuracy benchmark. This backup screenshot uses a generated color patch with the real evaluated model; it is deliberately not a real waste accuracy example. If the model is unavailable, present this screenshot instead of representing fixture output as real.

## 14. What this result does and does not establish

**Saksham Mighlani** — 40 seconds

Explain limitations plainly. White-background training can bias feature behavior. Unknown categories and multiple objects do not have reliable rejection guarantees. ECE is 0.1325, so probabilities are imperfect even after calibration. Exact historical evaluation source is not reconstructable because the run used an uncommitted tree. The preserved binary and predictions remain unchanged. Phase 2 needs an independent everyday-photo evaluation and a planned transfer-learning comparison.

## 15. Six contributions, one reproducible pipeline

**Shourya Bafna** — 35 seconds

Close by crediting each member. Om owns data and preprocessing, Aditya baseline descriptors and caching, Shourya SVM and integration review, Daksh forests, Prajjwal calibration and evaluation, and Saksham demo and submission assembly. Our main result is 0.8999 macro F1 with an uncertainty-aware educational prototype. Invite questions. The next slides are backup material only.

## 16. Backup: the selected SVM pipeline

**Shourya Bafna** — Backup

Explain C as the penalty for margin violations and gamma as the scale of the RBF similarity. Small gamma yields broad influence; larger gamma is more local. Square root and row L2 operate per image, so they do not fit statistics across train/validation/test. The selected 1608 columns are inside the saved pipeline, which expects a full 2760-vector. The extended winner has no HOG.

## 17. Backup: calibration is still imperfect

**Prajjwal Tripathi** — Backup

ECE is a weighted confidence-to-accuracy gap using 10 fixed bins. The chart shows only nonempty bins. Higher-confidence bins tend to be underconfident because observed accuracy exceeds average confidence. We have no saved paired uncalibrated test comparison, so do not claim a quantified test calibration improvement. Log loss and Brier score assess the full probability vector, while ECE summarizes confidence bins.

## 18. Backup: references and experiment records

**All members** — Backup

These sources motivate the methods. Project-specific metrics come from the versioned experiment CSV/JSON files, not external papers. The report contains the full bibliography. If asked about external baselines, explain that our split and duplicate review differ and we have not made a like-for-like performance claim against published work.
