---
marp: true
theme: default
paginate: true
header: "Smart Waste Segregation — Phase 1"
footer: "Team: Om Yadav, Aditya Bharadwaj, Shourya Bafna, Daksh Batra, Prajjwal Tripathi, Saksham Mighlani"
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 24px;
    padding: 40px;
  }
  h1 { color: #1a365d; }
  h2 { color: #2b6cb0; }
  .highlight { color: #2b6cb0; font-weight: bold; }
  .amber { color: #b7791f; font-weight: bold; }
  table { font-size: 20px; }
---

# Smart Waste Segregation
### Classical Machine Learning, Feature Engineering, and Calibrated Selective Classification

**Phase 1 Final Presentation**

**Contributors:**
- **Om Yadav**: Data Verification & Preprocessing
- **Aditya Bharadwaj**: Handcrafted Features & Caching
- **Shourya Bafna**: Training Protocol & SVM Baselines
- **Daksh Batra**: Random Forest & Feature Studies
- **Prajjwal Tripathi**: Confidence Calibration & Evaluation
- **Saksham Mighlani**: Demo Integration & Report Assembly

---

## 1. Problem Statement & Phase 1 Scope

- **Objective:** Automated, interpretable classification of recyclable municipal solid waste from digital imagery.
- **Phase 1 Constraint:** Strictly classical machine learning without deep neural networks or transfer learning.
- **Target Categories (6 classes):**
  - Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Key Challenges:**
  - Visual similarity across materials (clear plastic vs. clear glass; foil vs. metal cans).
  - High intra-class variability (crushed vs. intact items; color variations).
  - Risk of catastrophic evaluation leakage from identical physical objects.
  - Reliability requirement: automated sorting errors risk batch contamination.

---

## 2. Dataset Audit & Near-Duplicate Grouping

- **Source:** TrashNet dataset (2,527 raw digital photos).
- **Audit Findings & Exclusions:**
  - 6 identical files with conflicting class labels quarantined across 3 groups.
  - 2,521 usable images validated as 512×384 3-channel RGB.
- **Perceptual Hash Audit:**
  - 45 candidate pairs evaluated at Hamming distance $\le 6$.
  - 6 ambiguous pairs retained in conservative duplicate groups.
- **Group-Aware Stratified Splitting (70 / 15 / 15):**
  - **Train:** 1,764 images (69.97%)
  - **Validation:** 380 images (15.07%)
  - **Test:** 377 images (14.95%)
  - *Guarantee: No duplicate group spans across multiple splits.*

---

## 3. Preprocessing & Handcrafted Feature Engineering

### Preprocessing (Om Yadav)
- EXIF orientation correction; alpha channel compositing onto white.
- Aspect-preserving Lanczos letterboxing to 128×128 (baseline) and 256×256 (extended).

### Baseline Feature Vector: 1,806 Dimensions (Aditya Bharadwaj)
1. **HSV Color Histograms (32 dims):** 16 Hue, 8 Saturation, 8 Value; per-channel $L_1$-normalized.
2. **HOG (1,764 dims):** 9 orientation bins, 16×16 px/cell, 2×2 blocks, Dalal-Triggs $L_2$-Hys normalization.
3. **Uniform LBP (10 dims):** Circular $P=8, R=1.0$, uniform patterns, $L_1$-normalized.

### Opt-In Extended Spatial Vector: 2,760 Dimensions
- 6 spatial regions (global, foreground, 4 quadrants) × (multi-color space + multi-scale LBP).
- Winner pipeline selects 1,608 dimensions (global color + spatial texture).

---

## 4. Training Protocol & Model Selection

- **Harness (Shourya Bafna):** 5-fold `StratifiedGroupKFold` on outer training set (seed 42).
- **Leakage Prevention:** `StandardScaler` fitted strictly on 4/5 training complement per fold.

| Model Family | Descriptors | Dims | Mean CV Macro F1 | Fold Std |
| :--- | :--- | :---: | :---: | :---: |
| Dummy Baseline | Most Frequent | 1,806 | 0.0636 | 0.0002 |
| Linear SVM | Baseline (All) | 1,806 | 0.6007 | 0.0284 |
| RBF SVM | Baseline (All) | 1,806 | 0.6731 | 0.0339 |
| Random Forest | Color (HSV) | 32 | 0.7366 | 0.0260 |
| Random Forest | Baseline (All) | 1,806 | 0.7090 | 0.0522 |
| **Refined RF (Daksh)** | Ext. Spatial | 2,760 | 0.8370 | 0.0241 |
| **Extended RBF SVM (Shourya)** | Ext. Global/Spat | 1,608 | **0.8803** | **0.0271** |

*Selected Architecture: Extended RBF SVM ($C=10.0, \gamma=10.0$, root\_l2 normalization).*

---

## 5. Machine Learning Insights: SVM vs. Random Forest

### Random Forest Subspace Dilution (Daksh Batra)
- Baseline RF: 32 HSV color alone ($0.7366$) outperformed all 1,806 features ($0.7090$).
- *Reason:* Sampling $m = \sqrt{1806} \approx 42$ features per split meant color was only $1.8\%$ of candidates.
- *Remedy:* Tuning `max_features=0.25` on extended spatial descriptors restored RF to **0.8370 $\pm$ 0.0241**.

### SVM Nonlinear Boundary Advantage (Shourya Bafna)
- RBF kernel with elementwise square root + $L_2$ normalization (`root_l2`) mapped complex spatial color-texture interactions into linearly separable space.
- Winner 5-fold CV macro F1 reached **0.8803 $\pm$ 0.0271**, winning model selection.

---

## 6. Confidence Calibration & Abstention (Prajjwal Tripathi)

### Platt Sigmoid Calibration
- Conditioned on 5 frozen training folds via `CalibratedClassifierCV(ensemble=True)`.
- Base estimator trained on 4/5 complement; sigmoid calibrator fitted on 1/5 withheld fold.

### Validation-Only Abstention Selection
- 101-point threshold grid ($\tau \in [0.00, 1.00]$) evaluated solely on 380 validation images.
- **Selected Threshold $\tau = 0.52$**:
  - Validation Coverage: **98.16%** (373 / 380 accepted)
  - Validation Accepted Accuracy: **90.08%** (Target: $\ge 90\%$)
- *Limitation Note: Abstention filters low-confidence ambiguity; it does NOT reliably detect out-of-distribution unknown classes.*

---

## 7. Frozen Test Evaluation Results

Evaluated exactly once on the untouched test partition (377 images):

### Unconditional Performance (Before Abstention)
- **Macro F1:** **0.8999** (Target: $\ge 0.86$)
- **Accuracy:** **90.19%** (340 / 377 correct)
- **Multiclass Log Loss:** 0.4599
- **Multiclass Brier Score:** 0.1828
- **Expected Calibration Error (ECE):** 0.1325 (10 fixed bins)

### Selective Performance (After Abstention at $\tau = 0.52$)
- **Coverage:** **98.41%** (371 / 377 accepted)
- **Accepted Accuracy:** **91.37%** (339 / 371 correct, +1.19 percentage points)
- **Accepted Macro F1:** **0.9129** (+1.30% relative gain)

---

## 8. Test Confusion Matrix & Selective Error Breakdown

| True \ Pred | Cardboard | Glass | Metal | Paper | Plastic | Trash | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardboard** | **57** | 0 | 0 | 1 | 2 | 0 | 60 |
| **Glass** | 0 | **68** | 3 | 3 | 1 | 0 | 75 |
| **Metal** | 1 | 6 | **54** | 0 | 0 | 0 | 61 |
| **Paper** | 1 | 2 | 1 | **81** | 1 | 3 | 89 |
| **Plastic** | 1 | 4 | 3 | 2 | **61** | 1 | 72 |
| **Trash** | 1 | 0 | 0 | 0 | 0 | **19** | 20 |

### Breakdown of the 6 Abstained Images:
- 5 misclassifications were successfully caught and withheld!
- Only 1 correct sample was withheld (`paper348`, confidence 0.4784).
- Abstention directly improved user safety by eliminating 5 errors.

---

## 9. Interactive Gradio Demo (Saksham Mighlani)

- **Architecture:** Local web interface in `app.py` with process-level model reuse.
- **Low Overhead:** 66.3 MB evaluated model loaded and validated once; zero reload per request.
- **Shared Contract:** Calls `predict_image(image, artifact)` strictly without reimplementing logic.
- **Uncertainty UX:**
  - Displays primary prediction, calibrated confidence, and full 6-class table.
  - When $P_{\max} < 0.52$: prominent amber banner — *"Uncertain — manual inspection recommended."*
  - Tentative class shown separately without endorsement.
- **Privacy & Safety:** Local-only execution (`share=False`), photos not stored in project directories (Gradio uses transient OS temp storage during transport), memory-bounded against decompression bomb attacks ($>16$ MP).

---

## 10. Robustness, Testing & Submission Synthesis

### Automated Test Suite: 156 Passing, 1 Skipped
- Preprocessing & data verification: 32 tests (Om Yadav)
- Handcrafted features & caching: 16 tests (Aditya Bharadwaj)
- Training protocol & SVM: 34 tests (Shourya Bafna)
- Random Forest: 4 tests (Daksh Batra)
- Calibration & prediction contract: 18 tests (Prajjwal Tripathi)
- Demo UI & integration: 22 passed, 1 skipped for missing model binary (Saksham Mighlani)

### Submission Deliverables Assembled
- IEEE LaTeX Report (`report/main.tex`, `references.bib`, separate HTML preview `report.html`)
- Presentation source (`slides.md`) and interactive viewer (`index.html`)
- Clean-checkout reproduction guide and documentation
- Locked dependencies (`requirements.lock`, `requirements-demo.lock`)
- *Note: PDF compilation from LaTeX source remains an explicit outstanding check.*

---

## 11. Known Limitations & Phase 2 Outlook

1. **Historical Package Provenance Qualification:**
   - Author's account: pre-commit docstring/linter line-wrapping altered package hash from `26e7e076...` to `cab3b677...`.
   - The uncommitted state could not be independently reconstructed from Git history; recorded as author's account, not independently proven.
   - Evaluated model was preserved without refitting to maintain strict evaluation integrity.
2. **Out-of-Distribution Sensitivity:**
   - Confidence-based abstention filters visual ambiguity among the 6 classes; it does NOT guarantee rejection of unknown objects (e.g., electronic waste, organic food).
3. **Phase 2 Roadmap:**
   - Fine-tuned lightweight Vision Transformers / ConvNeXt for representation comparison.
   - Conformal prediction for finite-sample coverage guarantees.
   - On-device edge deployment and real-time conveyor stream processing.

---

# Questions & Viva Discussion

**Smart Waste Segregation Team**
*Thank you!*
