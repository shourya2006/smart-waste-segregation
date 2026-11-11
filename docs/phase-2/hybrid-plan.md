# Phase 2 Hybrid ML + DL Plan

**Status:** PRE-REGISTERED (Committed prior to running any hybrid training or evaluations)  
**Author:** Daksh Batra  
**Reviewer:** Shourya Bafna  
**Specification:** Classical ML on CNN embeddings, feature-level early fusion, and late probability fusion on the 5 frozen Phase 1 cross-validation folds.

---

## 1. Scientific Protocol & Rationale

### 1.1 Rationale for the Hybrid Approach
In Phase 1, classical handcrafted descriptors (HSV color histograms, HOG spatial shape gradients, and Local Binary Patterns) achieved strong performance across classes with distinct textures and colors (e.g., cardboard and paper). However, they exhibited systematic confusion across **glass, metal, and plastic**:
- Glass and plastic bottles share specular highlights, transparency, and smooth surface gradients.
- Metal cans and smooth plastics produce similar high-frequency edge responses under studio lighting.
- The Phase 1 confusion matrix revealed that metal-to-glass (24 images) and plastic-to-glass/metal were among the dominant error modes.

Pretrained deep convolutional neural networks (MobileNetV2 and EfficientNet-B0) extract hierarchical representations: early layers capture primitive edges/textures, while deeper layers capture global object geometry, contours, and semantic context.

**Hypothesis to test:** Combining CNN embeddings with handcrafted features (early fusion) or combining predictions from CNN and SVM models (late fusion) captures complementary information, improving overall classification and specifically reducing confusion among glass, metal, and plastic.

### 1.2 Strict Evaluation Rules
1. **Split Isolation:**
   - **Training CV Folds (1,764 images, 5 frozen folds in `docs/training/folds.json`):** Used exclusively for model fitting, hyperparameter selection, and out-of-fold (OOF) validation metrics.
   - **Validation Split (380 images):** Strictly held out from hybrid model selection, tuning, and comparisons. It is reserved exclusively for Prajjwal Tripathi to calibrate probability estimates and sweep abstention thresholds.
   - **Test Split (377 images):** Completely untouched and uninspected until the final single frozen system evaluation.
2. **Evaluation Metric:** Model ranking and selection are determined strictly by **mean Macro F1 across the 5 frozen training folds**, accompanied by sample standard deviation (`mean ± std`). We also track overall accuracy and per-class recall for all six classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
3. **No Data Leakage:** Every learned step (standardization scalers, L2 normalization, PCA if used, classifiers, probability calibrators, fusion weights) must be fitted strictly inside the training indices of each fold.
4. **Decision Rule:** A hybrid model is declared superior to single-paradigm baselines **only if its mean Macro F1 exceeds the best single model by more than the cross-fold standard deviation** ($\Delta > \sigma$). If the difference is within the fold standard deviation, we report that there is no statistically clear gain.
5. **Pre-Registration:** All candidates, grids, and weighting rules are registered here before running experiments. Any variant tested after inspecting initial runs must be explicitly marked as *"added after seeing results"*.

---

## 2. Reference Baselines

The hybrids will be compared against the best single models from Phase 1 and Phase 2:
1. **Phase 1 Reference Winner:**
   - Model: Extended RBF SVM (`C=10.0`, `gamma=10.0`, `root_l2` normalization).
   - Features: `global_color_spatial_texture` (1,608 selected columns from 2,760 extended descriptors).
   - Reported Performance: Mean CV Macro F1 = **0.8803 ± 0.0215** on the exact same 5 frozen folds.
2. **Phase 2 CNN Reference Baseline:**
   - MobileNetV2 (frozen backbone, Dropout=0.2, Linear head) from Aditya's ablation study / `waste-dl-train cv`.

---

## 3. Pre-Registered Hybrid Candidates

### 3.1 H1: Classical ML on Pretrained CNN Embeddings
We test whether classical maximum-margin classifiers (SVM) or regularized linear models (Logistic Regression) extract better class boundaries from frozen 1,280-dimensional CNN representations than standard SGD-trained linear heads on small sample sizes ($N=1,764$).

- **Backbones evaluated:**
  1. `mobilenet_v2` (1,280-dimensional frozen embeddings)
  2. `efficientnet_b0` (1,280-dimensional frozen embeddings)
- **Feature Preprocessing:**
  - Fold-local `StandardScaler`: mean and variance estimated strictly on fold training embeddings and applied to fold validation embeddings.
- **Classifiers & Hyperparameter Grids:**
  - **H1.1: Logistic Regression:**
    - Regularization: L2 penalty.
    - Solver: `lbfgs`.
    - $C \in \{0.01, 0.1, 1.0, 10.0\}$.
    - `max_iter`: 2000.
  - **H1.2: Linear SVM:**
    - `LinearSVC(dual='auto', random_state=42, max_iter=20000)`.
    - $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$.
  - **H1.3: Non-linear RBF SVM:**
    - `SVC(kernel='rbf', random_state=42, cache_size=512)`.
    - $C \in \{0.1, 1.0, 10.0, 100.0\}$.
    - $\gamma \in \{\text{'scale'}, 10^{-3}, 10^{-4}\}$.

### 3.2 H2: Early Feature Fusion (Handcrafted + CNN Embeddings)
We concatenate Phase 1's extended handcrafted feature block with the CNN embedding block to form a joint multi-modal representation.

- **Feature Inputs:**
  - **Handcrafted block:** Phase 1 extended features (`global_color_spatial_texture`), dimension $D_{\text{hc}} = 1,608$.
  - **Deep block:** Pretrained CNN embedding (`mobilenet_v2` or `efficientnet_b0`), dimension $D_{\text{cnn}} = 1,280$.
  - Total concatenated dimension: $D_{\text{total}} = 1,608 + 1,280 = 2,888$.
- **Per-Block Normalization and Dimensional Weighting:**
  To prevent the higher-dimensional handcrafted block or arbitrary feature magnitudes from dominating the RBF kernel distance metric, we enforce a strict fold-local normalization protocol:
  1. **Block-wise standard scaling:** Fit `StandardScaler` independently on each block using fold-training rows only.
  2. **Dimension equalization factor:** Divide each standardized block by $\sqrt{D_{\text{block}}}$:
     $$x_{\text{hc}}' = \frac{\text{StandardScaler}(x_{\text{hc}})}{\sqrt{1608}}, \quad x_{\text{cnn}}' = \frac{\text{StandardScaler}(x_{\text{cnn}})}{\sqrt{1280}}$$
     This ensures that the expected squared Euclidean norm of each block is 1, granting both feature modalities equal total geometric variance before concatenating:
     $$x_{\text{fused}} = [x_{\text{hc}}' \,\|\, x_{\text{cnn}}']$$
- **Classifiers & Hyperparameter Grids:**
  - **H2.1: RBF SVM on Fused Features (MobileNetV2):**
    - $C \in \{1.0, 10.0, 50.0, 100.0\}$.
    - $\gamma \in \{\text{'scale'}, 10^{-3}, 10^{-4}\}$.
  - **H2.2: RBF SVM on Fused Features (EfficientNet-B0):**
    - Same grid as H2.1.
  - **H2.3: Linear SVM on Fused Features:**
    - $C \in \{0.01, 0.1, 1.0, 10.0\}$.

### 3.3 H3: Late Probability Fusion (Pre-registered for PR B)
In PR B, we combine the out-of-fold probability vectors produced by the best deep CNN model ($p_{\text{cnn}}$) and the calibrated Phase 1 extended SVM ($p_{\text{svm}}$).

- **Formulation:**
  $$p_{\text{hybrid}} = w \cdot p_{\text{cnn}} + (1 - w) \cdot p_{\text{svm}}$$
  where $w \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$.
- **Nested Cross-Validation for Fusion Weight $w$:**
  To guarantee zero data leakage when evaluating H3:
  1. For each outer fold $k \in \{0, 1, 2, 3, 4\}$:
  2. Take the OOF probability predictions on the *other 4 folds* (the training portion of outer fold $k$).
  3. Sweep $w \in \{0, 0.1, \dots, 1\}$ and select $w_k^*$ that maximizes macro F1 across those 4 folds.
  4. Evaluate held-out fold $k$ using its assigned $w_k^*$.
  5. The held-out fold never influences its own fusion weight.

---

## 4. Required Deliverables for PR A

1. **`src/waste_sorting/hybrid.py`**:
   - `load_aligned_features(backbone, split="train", ...)`: Verifies identical image IDs and order between CNN embeddings and Phase 1 extended features.
   - `run_cv(candidate, ...)`: Fits candidate pipelines across the 5 frozen folds, tracking fold metrics, fit times, and per-class recall.
   - `phase1_svm_oof(...)`: Generates and exports out-of-fold calibrated probabilities for `reconstruct_selected_pipeline()` across the 5 training folds (1,764 rows, format matching CNN `oof_predictions.csv`).
   - CLI commands: `waste-hybrid cv` and `waste-hybrid phase1-oof`.
2. **`tests/test_hybrid.py`**:
   - Unit tests for ID alignment verification, fold-local scaling, dimensional block weighting, and OOF probability generation.
3. **`docs/phase-2/hybrid/`**:
   - Results table, glass/metal/plastic confusion analysis, and examiner viva notes.
