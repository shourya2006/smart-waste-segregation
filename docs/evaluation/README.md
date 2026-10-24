# Model Calibration, Validation Abstention, and Test Evaluation

This document records the frozen calibration, threshold selection, and final test
evaluation of the Phase 1 Smart Waste Segregation model.

## 1. Frozen Model and Calibration Protocol

- **Selected Model**: Extended RBF SVM (`global_color_spatial_texture`)
- **Feature Extraction**: 2,760 extended descriptors; 1,608 internally selected
- **Normalization**: Elementwise square root then per-image L2 (`root_l2`)
- **SVM Hyperparameters**: `C=10.0`, `gamma=10.0`, `class_weight=None`, `seed=42`
- **Calibration Method**: Sigmoid calibration via `CalibratedClassifierCV(ensemble=True)`
- **Calibration Folds**: 5 explicit frozen folds from `docs/training/folds.json`
- **Training Complement Isolation**: Each base pipeline is trained strictly on 4/5 of outer training data; each calibrator is fitted on the withheld 1/5. No model is calibrated on data it trained on.
- **Limitation Note**: Chosen hyperparameters were previously selected using these training folds; calibration diagnostics on those rows are not independent performance estimates.

## 2. Validation Abstention Selection

- **Validation Split**: 380 images
- **Threshold Grid**: 0.00 through 1.00 (step 0.01)
- **Selected Threshold**: **0.52**
- **Validation Coverage**: **0.9816** (373/380)
- **Validation Accepted Accuracy**: **0.9008**
- **Target Met**: `True` (Target: accuracy ≥90%, coverage ≥80%)
- **Selection Rule**: highest coverage with accuracy >= target and coverage >= min_coverage; ties to lower threshold

> [!NOTE]
> Abstention does not establish reliable detection of unknown object categories.

## 3. Untouched Test Set Evaluation

- **Test Split**: 377 images

### Unconditional Performance (Before Abstention)

- **Macro F1**: **0.8999**
- **Accuracy**: **0.9019** (340/377)
- **Multiclass Log Loss**: **0.4599**
  - Formula: `- (1 / N) * sum_{i=1}^N sum_{c=1}^C y_{ic} * ln(p_{ic})`
- **Multiclass Brier Score**: **0.1828**
  - Formula: `(1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2`
- **Expected Calibration Error (ECE)**: **0.1325** (10 fixed bins)
  - Formula: `sum_{m=1}^{10} (|B_m| / N) * |acc(B_m) - conf(B_m)|`

### Per-Class Test Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `cardboard` | 0.9344 | 0.9500 | 0.9421 | 60 |
| `glass` | 0.8500 | 0.9067 | 0.8774 | 75 |
| `metal` | 0.8852 | 0.8852 | 0.8852 | 61 |
| `paper` | 0.9310 | 0.9101 | 0.9205 | 89 |
| `plastic` | 0.9385 | 0.8472 | 0.8905 | 72 |
| `trash` | 0.8261 | 0.9500 | 0.8837 | 20 |

### Confusion Matrix

| True \ Pred | `cardboard` | `glass` | `metal` | `paper` | `plastic` | `trash` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cardboard` | 57 | 0 | 0 | 1 | 2 | 0 |
| `glass` | 0 | 68 | 3 | 3 | 1 | 0 |
| `metal` | 1 | 6 | 54 | 0 | 0 | 0 |
| `paper` | 1 | 2 | 1 | 81 | 1 | 3 |
| `plastic` | 1 | 4 | 3 | 2 | 61 | 1 |
| `trash` | 1 | 0 | 0 | 0 | 0 | 19 |

### Selective Performance (After Abstention at Frozen Threshold)

- **Threshold**: `0.52`
- **Accepted Examples**: 371 / 377
- **Abstained Examples**: 6 / 377
- **Coverage**: **0.9841** (98.4%)
- **Accepted Accuracy**: **0.9137** (91.4%, +1.19 percentage points over unconditional)
- **Accepted Macro F1**: **0.9129**

## 4. Visualizations

![Confusion Matrix](confusion-matrix.png)

![Reliability Diagram](reliability-diagram.png)

## 5. Artifacts and Reproduction

Reproduction command:
```sh
python -m waste_sorting.evaluation --output docs/evaluation --artifacts artifacts/calibrated_svm
```
