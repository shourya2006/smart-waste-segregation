# Prajjwal Tripathi: Confidence Calibration, Abstention, and Model Evaluation

Phase 1 member notes for Prajjwal Tripathi.
This document details the statistical design, calibration methodology, validation threshold
selection, frozen test evaluation, failure modes, reproduction instructions, and verified metrics.

## 1. Implemented Contribution

- `src/waste_sorting/evaluation.py`:
  - Data validation and feature extraction preserving audited outer splits (1,764 train, 380 val, 377 test)
    with image checksum verification against `docs/dataset/manifest.csv`.
  - Reconstruction of the winning extended RBF SVM pipeline (`global_color_spatial_texture`, 1,608 selected
    descriptors from 2,760 input, elementwise square root + L2 normalization, C=10, gamma=10, seed=42).
  - Group-aware sigmoid calibration using `CalibratedClassifierCV(ensemble=True)` with the 5 explicit
    frozen training folds from `docs/training/folds.json`.
  - Rigorous training-complement isolation: base estimators fit only on 4/5 training complement; Platt
    calibrators fit only on the 1/5 withheld training fold rows. No estimator calibrates on data it trained on.
  - Threshold sweep (0.00 to 1.00 in increments of 0.01) and deterministic threshold selection on validation data only.
  - Guard preventing accidental overwrites of existing final test evaluation records.
  - Comprehensive multiclass evaluation: accuracy, macro F1, per-class metrics, confusion matrix, multiclass
    log loss, multiclass Brier score, Expected Calibration Error (ECE) with 10 fixed confidence bins, and
    selective metrics under abstention.
  - Automated visualization generation: confusion matrix heatmap and reliability diagram.
- `src/waste_sorting/prediction.py`:
  - Shared inference interface: `predict_image(image, artifact)`.
  - Full input validation (PIL Image or path; checks artifact metadata, dimensions, classes, and threshold).
  - Returns `PredictionResult` with `predicted_class`, `probabilities` mapped to class order, `max_probability`,
    and `abstained` flag.
  - Rejects incompatible extractor inputs and prevents substitution of the baseline 1,806-feature vector.
- `tests/test_evaluation.py` & `tests/test_prediction.py`:
  - 18 unit tests verifying calibration fold isolation, exclusion of validation/test data from fitting,
    probability properties (shape, finiteness, sum-to-one), threshold selection tie-breakers and fallback logic,
    boundary equality, prediction consistency after serialization, artifact validation, and overwrite guard.
- `docs/evaluation/`:
  - `README.md`: Consolidated technical evaluation report.
  - `summary.json`: Machine-readable summary with complete provenance and metrics.
  - `validation-threshold-sweep.csv`: Complete 101-point threshold sweep on validation data.
  - `validation-metrics.json`: Unconditional and thresholded validation metrics.
  - `test-predictions.csv`: Sample-by-sample predictions and probabilities for all 377 test images.
  - `test-metrics.json`: Final test set metrics.
  - `confusion-matrix.png` & `reliability-diagram.png`: Publication-ready visual artifacts.

## 2. Statistical Protocol & Scopes

1. **Outer Split Protection**:
   - Total dataset: 2,521 included images across 6 classes.
   - Train: 1,764 images (70%).
   - Validation: 380 images (15%).
   - Test: 377 images (15%).
   - Duplicate groups stay strictly within their outer split.
   - Model fitting and probability calibration use outer training data only. Outer validation is used
     solely for threshold selection; the test set is evaluated frozen exactly once.
2. **Calibration Protocol**:
   - Model: Extended RBF SVM (`C=10.0, gamma=10.0, class_weight=None, seed=42`).
   - Normalization: Elementwise square root then per-image L2 (`root_l2`).
   - Method: Platt scaling (sigmoid calibration) via `CalibratedClassifierCV(ensemble=True)`.
   - CV Folds: 5 frozen group-aware folds (`StratifiedGroupKFold`) loaded from `docs/training/folds.json`.
   - Fold Isolation: Base pipeline $k$ trains on training complement ($\sim 1,411$ images); calibrator $k$
     fits on withheld training fold ($\sim 353$ images).
   - Limitation: Hyperparameters C=10, gamma=10 were previously chosen using these folds; calibration
     diagnostics on those rows are not independent performance estimates.
3. **Abstention Selection Protocol (Validation Only)**:
   - Evaluated over a 101-point grid: $\tau \in [0.00, 1.00]$ with step $0.01$.
   - Accept rule: $\max_c P(y=c|x) \ge \tau$.
   - Target criteria: Accepted accuracy $\ge 90\%$, Coverage $\ge 80\%$.
   - Primary selection: Threshold with highest coverage satisfying both conditions; ties broken by lower threshold.
   - Fallback rule: Highest accepted accuracy among coverage $\ge 80\%$; ties broken by higher coverage, then lower threshold.
   - Limitation: Abstention does not establish reliable detection of unknown or out-of-distribution object categories.
4. **Frozen Final Test Evaluation**:
   - Evaluated on 377 untouched test images using the frozen calibrated ensemble and validation-selected threshold.
   - Predictions saved to `docs/evaluation/test-predictions.csv` for independent inspection without refitting.

## 3. Mathematical Formulas

- **Coverage**:
  $$\text{Coverage}(\tau) = \frac{N_{\text{accepted}}(\tau)}{N_{\text{total}}}$$

- **Accepted Accuracy**:
  $$\text{Accuracy}_{\text{accepted}}(\tau) = \frac{\sum_{i \in \text{accepted}} \mathbf{1}(\hat{y}_i = y_i)}{N_{\text{accepted}}(\tau)}$$

- **Multiclass Log Loss (Cross-Entropy)**:
  $$L_{\text{log}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln(p_{ic})$$
  where $y_{ic} \in \{0, 1\}$ is the one-hot indicator, $p_{ic} = P(Y=c|X_i)$ clipped to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-15}$.

- **Multiclass Brier Score**:
  $$BS = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$
  Sum of squared differences between predicted calibrated probability distribution and one-hot true label vector, averaged over all $N$ samples.

- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  where confidence space $[0, 1]$ is partitioned into 10 fixed intervals $[0.0, 0.1), \dots, [0.9, 1.0]$. Bins with $|B_m| = 0$ contribute 0.

## 4. Verified Results

### Validation Abstention Selection (380 images)
- **Selected Threshold**: **0.52**
- **Coverage**: **98.16%** (373 / 380 images accepted)
- **Accepted Accuracy**: **90.08%** (336 / 373 correct)
- **Abstained Count**: 7 images
- **Target Met**: True (exceeds $\ge 90\%$ accuracy and $\ge 80\%$ coverage)

### Untouched Test Set Performance (377 images)

#### Unconditional (Before Abstention)
- **Macro F1**: **0.8999** (Target: $\ge 0.86$)
- **Accuracy**: **0.9019** (340 / 377 correct)
- **Multiclass Log Loss**: **0.4599**
- **Multiclass Brier Score**: **0.1828**
- **Expected Calibration Error (ECE)**: **0.1325**

#### Per-Class Breakdown (Test)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `cardboard` | 0.9344 | 0.9500 | 0.9421 | 60 |
| `glass` | 0.8500 | 0.9067 | 0.8774 | 75 |
| `metal` | 0.8852 | 0.8852 | 0.8852 | 61 |
| `paper` | 0.9310 | 0.9101 | 0.9205 | 89 |
| `plastic` | 0.9385 | 0.8472 | 0.8905 | 72 |
| `trash` | 0.8261 | 0.9500 | 0.8837 | 20 |
| **Macro Average** | **0.8942** | **0.9082** | **0.8999** | **377** |

#### Confusion Matrix (Test)
```
True \ Pred   cardboard  glass  metal  paper  plastic  trash
cardboard            57      0      0      1        2      0
glass                 0     68      3      3        1      0
metal                 1      6     54      0        0      0
paper                 1      2      1     81        1      3
plastic               1      4      3      2       61      1
trash                 1      0      0      0        0     19
```

#### Selective Performance (At Frozen Threshold $\tau = 0.52$)
- **Threshold**: 0.52
- **Coverage**: **98.41%** (371 accepted / 377 total)
- **Abstained Count**: 6 images (1.59% abstention rate)
- **Accepted Accuracy**: **91.37%** (339 / 371 correct, +1.19 percentage points over unconditional)
- **Accepted Macro F1**: **0.9129** (+1.30% gain over unconditional)

## 5. Artifact and Provenance Fingerprints

- **Source Git Revision**: `8c6f035f21a5e83c21270b897f5c449a023bf0aa`
- **Python Version**: 3.12.5
- **Platform**: macOS-26.6.1-arm64
- **Manifest SHA-256**: `affd3e9d29e612b0179820a2a20ed21395e26d89a094f02bb7390d9400563314`
- **Folds SHA-256**: `6525c05675d2c9470659f2ad6a5721cadbe8558ac977ef7ef06dc858488b6625`
- **Base Training Fingerprint**: `8b721ac54a69b6dfd39410ebec83caa9093ccad6747e8312e5e38dd1b1c572d5`
- **Requirements Lock SHA-256**: `56ecb8a1355df784c856df7472bc5252309d4d1b32a5b7f4f78a7cb182f8cd52`
- **Pinned Key Packages**:
  - `numpy`: 2.5.4
  - `scipy`: 1.18.1
  - `scikit-learn`: 1.9.1
  - `scikit-image`: 0.26.0
  - `pandas`: 3.0.6
  - `joblib`: 1.6.0
  - `matplotlib`: 3.11.2
  - `seaborn`: 0.13.2
  - `pillow`: 13.0.0

## 6. Exact Reproduction Commands

```sh
# 1. Run all checks and tests
python -m unittest discover -s tests -v
ruff check .

# 2. Run calibration, validation abstention selection, and test evaluation
python -m waste_sorting.evaluation --output docs/evaluation --artifacts artifacts/calibrated_svm --n-jobs 4

# 3. Test the prediction interface
python -c "
from waste_sorting.prediction import predict_image
res = predict_image('data/raw/trashnet/cardboard/cardboard1.jpg', 'artifacts/calibrated_svm/model.joblib')
print(res)
"
```
