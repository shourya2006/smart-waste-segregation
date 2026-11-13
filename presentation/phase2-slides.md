# Phase 2 — Smart Waste Segregation

## Smart Waste Segregation | Phase 2

- Deep transfer learning • Hybrid fusion • Frozen evaluation
- CSAI Project 12 | 13 November 2026
- Om Yadav · Aditya Bharadwaj · Shourya Bafna · Daksh Batra · Prajjwal Tripathi · Saksham Miglani

## Problem & research questions

- Six TrashNet classes: cardboard, glass, metal, paper, plastic, trash
- Can pretrained CNNs outperform handcrafted features?
- Can hybrid fusion reduce complementary errors?
- Does training-OOF calibration transfer to held-out test?

## Frozen data protocol

- 2,521 images: 1,764 train / 380 validation / 377 test
- Group-aware split; five fixed training folds; seed 42
- All model selection and thresholds fixed before one test pass
- No external robustness dataset collected

## Data pipeline & regularisation

- 224×224 letterbox; ImageNet normalisation
- Training-only crop, flip, rotation and colour jitter
- Validation/test use deterministic transforms
- Class weighting and balanced sampling evaluated separately

## Transfer learning backbones

- MobileNetV2 and EfficientNet-B0 pretrained on ImageNet
- Frozen baseline: train only six-class linear head
- A2.1: unfreeze final two MobileNetV2 blocks
- 893,766 trainable parameters; differential learning rates

## Ablation: why A2.1?

- 13 configurations × 5 folds = 65 fold evaluations
- BASE macro F1: 0.8515 ± 0.0235
- A2.1 macro F1: 0.8825 ± 0.0219
- A2.1 selected for the final CNN and H3 hybrid

## Early vs late hybrid fusion

- H2: concatenated CNN embeddings + handcrafted features + RBF SVM
- H2 cross-validation macro F1: 0.9044 ± 0.0332
- H3: 50% CNN + 50% SVM probability blending
- Nested fold selection; H3 CV macro F1: 0.9364 ± 0.0171

## Calibration & abstention

- Temperatures fitted only on training out-of-fold predictions
- Frozen demo thresholds: SVM 0.76 • A2.1 0.87 • H3 0.53
- Below threshold: uncertain — manual inspection recommended
- Demo is educational, not a real-world disposal authority

## Untouched test: model comparison

- Phase 1 refit SVM: macro F1 0.8963; accuracy 90.19%
- A2.1 CNN: macro F1 0.8577; accuracy 88.06%
- H3 hybrid: macro F1 0.9233; accuracy 93.63%
- All results from the same 377-image frozen test split

## Statistical evidence

- Exact McNemar: H3 vs A2.1, p = 0.000019 (significant)
- H3 vs Phase 1 SVM, p = 0.0351 (not Bonferroni-significant)
- Bonferroni alpha = 0.0167 for three comparisons
- Avoid claiming H3 statistically beats SVM after correction

## Test confusion matrices

- H3 hybrid correct on 353 / 377 images
- H3 improves glass/metal confusion vs single models
- Trash class has only 20 test examples
- Error analysis and confusion figures are saved with frozen evaluation

## Grad-CAM: what it shows

- 12 inspected A2.1 cases: six correct and six errors
- 11/12 show ≥80% foreground activation (heuristic mask)
- Glass, metal and plastic still confuse under reflections
- Small sample cannot rule out background dependence

## Live demo workflow

- Upload image and select H3, A2.1 or Phase 1 SVM
- View calibrated class probabilities and predicted label
- Frozen threshold triggers uncertainty banner
- SHA-256 checks reject missing or altered model artifacts

## Limitations & conclusion

- H3 highest observed test macro F1: 0.9233
- Selection on reused folds may make CV optimistic
- TrashNet white-background bias; no external robustness set
- Frozen protocol supports reproducible, auditable evaluation
