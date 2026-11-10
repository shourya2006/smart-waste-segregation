# Phase 2 Ablation Plan: Deep Learning Backbones & Regularisation

**Status:** PRE-REGISTERED (Committed prior to running any ablation experiments)  
**Author:** Aditya Bhardwaj  
**Reviewer:** Om Yadav  
**Specification:** Controlled one-factor-at-a-time (OFAT) ablation study on the frozen Phase 1 cross-validation folds.

---

## 1. Scientific Protocol & Integrity Rules

1. **Pre-Registration:** This plan defines the complete hypothesis space, candidate configurations, evaluation metric, and validation split protocol *before* any Phase 2 model weights are trained or compared.
2. **Selection Rule:** Model selection and configuration ranking are strictly governed by **mean Macro F1 across the 5 frozen training cross-validation folds** (`docs/training/folds.json`).
3. **Split Isolation:**
   - **Training CV Folds (1,764 images, 5 folds):** Used exclusively for training candidate models and computing out-of-fold validation metrics for the ablation table.
   - **Validation Split (380 images):** Strictly held out from model selection and architecture comparisons. It is reserved exclusively for Prajjwal Tripathi to calibrate probability estimates (temperature scaling / Platt scaling) and sweep abstention thresholds.
   - **Test Split (377 images):** Completely untouched and uninspected until the final single frozen system evaluation.
4. **Honest Reporting:** If any additional configuration is evaluated that was not explicitly listed in this initial protocol, it must be explicitly labelled in all tables and reports as *"added after seeing results"*.
5. **Statistical Significance & Uncertainty:** We report both the mean and sample standard deviation across all 5 folds (`mean ± std`). If the performance difference between two variants is smaller than the cross-fold standard deviation ($\Delta < \sigma$), we do not claim superiority.

---

## 2. Baseline Configuration

The baseline is designed as a conservative, lightweight transfer-learning pipeline:

| Component | Baseline Setting | Rationale |
| :--- | :--- | :--- |
| **Backbone Architecture** | `mobilenet_v2` | Low parameter count ($\approx 3.5\text{M}$ total, $2.2\text{M}$ backbone), fast training. |
| **Weights** | `IMAGENET1K_V2` | High-quality modern torchvision pretrained weights. |
| **Backbone Mode** | Frozen | Feature extractor completely frozen; train classification head only. |
| **Input Resolution** | $224 \times 224$ | Standard ImageNet resolution; matches Phase 1 letterbox size. |
| **Augmentation** | Enabled (All 4 steps active) | Conservative label-preserving: RandomResizedCrop (scale 0.8-1.0), HorizontalFlip ($p=0.5$), Rotation ($\pm 10^\circ$), ColorJitter ($0.15/0.15/0.10$). |
| **Classification Head** | Linear head with Dropout ($p=0.2$) | `nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 6))`. |
| **Loss Function** | Standard Cross-Entropy (`nn.CrossEntropyLoss`) | No class weighting by default. |
| **Sampling** | Standard Shuffle (`shuffle=True`) | Uniform sampling from training fold. |
| **Batch Size & Optimizer** | Batch size 32, AdamW, weight decay $10^{-2}$ | Standard regularized optimization for transfer learning. |
| **Learning Rate & Schedule** | Initial LR $10^{-3}$, Cosine Annealing | Decays smoothly to $10^{-5}$ over training epochs. |
| **Epochs & Early Stopping** | Max 40 epochs, patience 8 on fold-validation loss | Prevents overfitting while allowing convergence. |
| **Random Seed** | 42 | Worker and PyTorch RNG seeded deterministically. |

---

## 3. One-Factor-At-A-Time (OFAT) Ablation Factors

Each experiment varies exactly one factor relative to the baseline configuration:

### Factor 1: Backbone Architecture
- **Variant 1.0 (Baseline):** `mobilenet_v2` (1,280-dim embedding, $300\text{ MFLOPs}$).
- **Variant 1.1:** `efficientnet_b0` (1,280-dim embedding, $390\text{ MFLOPs}$, compound scaling with SE attention).

### Factor 2: Fine-Tuning Strategy (Frozen vs. Partial Fine-Tuning)
- **Variant 2.0 (Baseline):** Frozen backbone (`freeze(model)`). Zero trainable backbone parameters; only the classification head ($\approx 7,680$ parameters) is trained.
- **Variant 2.1 (Partial fine-tuning, $n=2$):** Unfreeze top structural blocks (`unfreeze_last_blocks(model, n=2)`).
  - For `mobilenet_v2`: Unfreezes the last InvertedResidual block and final $1 \times 1$ conv (`features[17:19]`), adding **886,080** trainable parameters (39.8% of backbone).
  - For `efficientnet_b0`: Unfreezes stage 7 MBConv and final conv (`features[7:9]`), adding **1,129,392** trainable parameters (28.2% of backbone).
  - Classification head is trained with learning rate $10^{-3}$; unfrozen backbone layers use a differential lower learning rate $10^{-4}$.
  - Note: Train loops must call `keep_frozen_bn_eval(model)` after `model.train()` so frozen layers preserve ImageNet BatchNorm statistics.
- *(Reference parameter counts for other $n$: MobileNetV2 $n=1$: 412,160 / $n=3$: 1,206,080; EfficientNet-B0 $n=1$: 412,160 / $n=3$: 3,155,740).*

### Factor 3: Input Resolution
- **Variant 3.0 (Baseline):** $224 \times 224$ pixels.
- **Variant 3.1:** $160 \times 160$ pixels (faster edge inference, test latency vs accuracy tradeoff).

### Factor 4: Head Regularisation (Dropout)
- **Variant 4.0:** Dropout $p = 0.0$ (No dropout before linear head).
- **Variant 4.1 (Baseline):** Dropout $p = 0.2$.
- **Variant 4.2:** Dropout $p = 0.5$ (Aggressive regularisation).

### Factor 5: Data Augmentation Components
- **Variant 5.0 (Baseline):** Full augmentation (Crop, Flip, Rotation, Jitter enabled).
- **Variant 5.1:** No augmentation (`dl.augmentation.enabled = false` / all steps off).
- **Variant 5.2:** Without Random Resized Crop (`random_resized_crop = false`).
- **Variant 5.3:** Without Horizontal Flip (`horizontal_flip = false`).
- **Variant 5.4:** Without Random Rotation (`rotation = false`).
- **Variant 5.5:** Without Color Jitter (`color_jitter = false`).

### Factor 6: Class Imbalance Handling
- **Variant 6.0 (Baseline):** None (Uniform sampling, unweighted loss).
- **Variant 6.1:** Class-Weighted Cross-Entropy Loss (`class_weights(labels, num_classes)` computed strictly on fold-training labels).
- **Variant 6.2:** Balanced Random Sampling (`WeightedRandomSampler` via `make_balanced_sampler`).

---

## 4. Planned Results Table Schema

Each run logs individual fold metrics to `docs/phase-2/ablations/runs.csv`, which are aggregated into `docs/phase-2/ablations/summary.csv`:

| Exp ID | Factor Tested | Configuration Description | Mean Macro F1 | Std F1 | Mean Accuracy | Cardboard Recall | Glass Recall | Metal Recall | Paper Recall | Plastic Recall | Trash Recall | Epochs (Mean) | Mean Train Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASE** | Baseline | MobileNetV2, Frozen, 224, Drop 0.2, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A1.1** | Architecture | EfficientNet-B0, Frozen, 224, Drop 0.2, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A2.1** | Fine-Tuning | MobileNetV2, Last 2 Blocks, 224, Drop 0.2, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A3.1** | Resolution | MobileNetV2, Frozen, 160, Drop 0.2, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A4.0** | Dropout | MobileNetV2, Frozen, 224, Drop 0.0, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A4.2** | Dropout | MobileNetV2, Frozen, 224, Drop 0.5, Aug Full, None | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A5.1** | Augmentation | No Augmentation (All Off) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A5.2** | Augmentation | No Random Resized Crop | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A5.3** | Augmentation | No Horizontal Flip | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A5.4** | Augmentation | No Random Rotation | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A5.5** | Augmentation | No Color Jitter | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A6.1** | Imbalance | Class-Weighted Cross-Entropy | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **A6.2** | Imbalance | Balanced Random Sampling | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Execution Rules for PR B (Phase 2 Hand-off)

1. PR B will be launched only after Shourya Bafna's training loop PR is merged into `main`.
2. PR B will reuse Shourya's training engine directly to run all 13 planned configurations across the 5 CV folds (a total of 65 training runs).
3. All code modifications will be strictly confined to `scripts/run_ablations.py` (or CLI flags) and `docs/phase-2/ablations/*`. No model weights, checkpoints, or `.npz` caches will be committed to the repository.

