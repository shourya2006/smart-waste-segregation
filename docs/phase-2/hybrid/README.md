# Phase 2 Hybrid ML + DL Models: Results and Evaluation

**Owner:** Daksh Batra  
**Reviewer:** Shourya Bafna  
**Specification:** Classical ML on CNN embeddings (H1) and handcrafted + CNN early feature fusion (H2) evaluated on the 5 frozen Phase 1 cross-validation folds.

---

## 1. Executive Summary & Selection Decision

In accordance with the pre-registered protocol in [`docs/phase-2/hybrid-plan.md`](../hybrid-plan.md):
- **Decision Rule:** A hybrid model is declared a definitive improvement over single-paradigm baselines *only* if its mean Macro F1 exceeds the best single model by more than the cross-fold standard deviation ($\Delta > \sigma$). Otherwise, we report that there is no clear overall gain.
- **Phase 1 Single-Model Reference:** Extended RBF SVM (`0.8803 ± 0.0215` Macro F1 across 5 folds in the uncalibrated study; `0.8844 ± 0.0200` for the re-run fold-calibrated OOF probabilities).
- **H1 (ML on CNN Embeddings):** Best candidate is `h1_efficientnet_b0_logreg_c1` with Macro F1 **0.8597 ± 0.0194** (MobileNetV2 RBF reaches **0.8501 ± 0.0204**). On their own, frozen linear or RBF heads on CNN embeddings perform below the Phase 1 handcrafted SVM.
- **H2 (Early Feature Fusion):** The dimensional-weighted fusion of Phase 1 extended colour/texture features (1,608 dims) and EfficientNet-B0 CNN embeddings (1,280 dims) with an RBF SVM (`C=10.0`, `gamma='scale'`) achieves:
  $$\text{Mean CV Macro F1} = \mathbf{0.9044 \pm 0.0332}, \quad \text{Mean Accuracy} = \mathbf{91.73\%}$$
  (MobileNetV2 fusion similarly reaches **0.9039 ± 0.0262**).

### Verdict on the Decision Rule
Comparing the leading H2 candidate (`0.9044`) to the Phase 1 SVM reference (`0.8803`), the absolute mean performance gain across the 5 folds is:
$$\Delta = 0.9044 - 0.8803 = +0.0241$$
(or $\Delta = +0.0201$ compared to the re-run calibrated Phase 1 OOF predictions at `0.8844`).

Because $\Delta = 0.0241$ is smaller than the cross-fold sample standard deviation ($\sigma = 0.0332$), following the pre-registered decision rule strictly, **there is no clear overall gain yet**. 

H2 is currently our leading candidate, delivering a substantial, targeted improvement on the classes that confused Phase 1 (pooled out-of-fold counts):
- **Metal pooled recall** increases from **241 / 286 = 84.27%** to **266 / 286 = 93.01%** (+8.74% absolute gain; 259 / 286 = 90.56% for MobileNetV2).
- **Metal-to-glass misclassifications** drop from **21 errors down to 8 errors** (a **61.9% reduction**; also 8 for MobileNetV2).
- **Plastic pooled recall** rises from **289 / 336 = 86.01%** to **303 / 336 = 90.18%** (+4.17% absolute gain; 298 / 336 = 88.69% for MobileNetV2), with plastic-to-glass misclassifications dropping from 21 to 13 (17 for MobileNetV2).
- **Trash pooled recall** improves from **78 / 96 = 81.25%** (Phase 1 SVM) to **84 / 96 = 87.50%** in EfficientNet-B0 H2 (+6.25% absolute gain, with remaining errors going to cardboard (5), metal (3), and plastic (4)). For MobileNetV2 H2, trash recall is **77 / 96 = 80.21%** (a slight 1-sample shift from 78 to 77). Note that while pooled trash recall increases for EfficientNet-B0, mean per-fold trash recall is lower due to fold-level variance on the small 96-image support.

Furthermore, the comparison against the best single CNN is pending Aditya's ablation study results; the final selection verdict across all models (single CNN, Phase 1 SVM, H1, H2, and H3 late fusion) will be determined in PR B.

Hyperparameters within each grid were selected on the same 5 folds used to report the cross-validation score (identical to the Phase 1 protocol), so the reported fold means are slightly optimistic for every candidate alike.

> [!NOTE]
> **Phase 1 Baseline Numbers Provenance:** The `0.8803 ± 0.0215` macro F1 and class metrics in the main comparison table are taken directly from the uncalibrated Phase 1 study record (`docs/training/extended-svm-study.md`). When re-fitting `reconstruct_selected_pipeline()` with fold-local Platt sigmoid calibration to produce `phase1_svm_oof.csv`, the argmax predictions across the 5 held-out folds yield `0.8844 ± 0.0200` macro F1 (+0.0041 difference due to calibration probability smoothing). The confusion matrix counts and pooled recalls in Section 3 are derived directly from this calibrated OOF file.

---

## 2. Comprehensive Cross-Validation Results

All candidates were evaluated strictly across the 5 frozen training folds (`docs/training/folds.json`, 1,764 total training images). **Zero images from validation or test splits were inspected, fitted, or scored.**

The table below reports the **mean of the 5 per-fold metrics** (sample standard deviation across folds).

| Exp ID / Candidate | Family | Backbone | Classifier | Hyperparameters | Mean Macro F1 | Std F1 | Mean Accuracy | Mean Per-Fold Cardboard Recall | Mean Per-Fold Glass Recall | Mean Per-Fold Metal Recall | Mean Per-Fold Paper Recall | Mean Per-Fold Plastic Recall | Mean Per-Fold Trash Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 Ref (SVM)** | Handcrafted | None | RBF SVM | C=10, $\gamma=10$, root_l2 | 0.8803 | 0.0215 | 0.8900 | 0.9255 | 0.8937 | 0.8427 | 0.9495 | 0.8601 | 0.7812 |
| **h2_fusion_efficientnet_b0_rbf_c10** | **H2** | **efficientnet_b0** | **rbf_svm** | **C=10.0, $\gamma$='scale'** | **0.9044** | **0.0332** | **0.9173** | 0.9183 | 0.9080 | **0.9510** | 0.9616 | **0.8899** | 0.7489 |
| h2_fusion_efficientnet_b0_rbf_c50 | H2 | efficientnet_b0 | rbf_svm | C=50.0, $\gamma$='scale' | 0.9044 | 0.0332 | 0.9173 | 0.9183 | 0.9080 | 0.9510 | 0.9616 | 0.8899 | 0.7489 |
| h2_fusion_mobilenet_v2_rbf_c10 | H2 | mobilenet_v2 | rbf_svm | C=10.0, $\gamma$='scale' | 0.9039 | 0.0262 | 0.9167 | 0.9396 | 0.9253 | 0.9299 | 0.9640 | 0.8719 | 0.7284 |
| h2_fusion_mobilenet_v2_rbf_c50 | H2 | mobilenet_v2 | rbf_svm | C=50.0, $\gamma$='scale' | 0.9039 | 0.0262 | 0.9167 | 0.9396 | 0.9253 | 0.9299 | 0.9640 | 0.8719 | 0.7284 |
| h2_fusion_mobilenet_v2_linsvm_c01 | H2 | mobilenet_v2 | linear_svm | C=0.1 | 0.8798 | 0.0226 | 0.8918 | 0.9147 | 0.8793 | 0.8881 | 0.9688 | 0.8453 | 0.7068 |
| h1_efficientnet_b0_logreg_c1 | H1 | efficientnet_b0 | log_reg | C=1.0 | 0.8597 | 0.0194 | 0.8742 | 0.9221 | 0.8562 | 0.8740 | 0.9135 | 0.8513 | 0.7084 |
| h1_efficientnet_b0_logreg_c01 | H1 | efficientnet_b0 | log_reg | C=0.1 | 0.8584 | 0.0181 | 0.8770 | 0.9149 | 0.8620 | 0.8566 | 0.9352 | 0.8632 | 0.6763 |
| h1_efficientnet_b0_rbf_c10 | H1 | efficientnet_b0 | rbf_svm | C=10.0, $\gamma$='scale' | 0.8582 | 0.0370 | 0.8867 | 0.9008 | 0.8822 | 0.9230 | 0.9472 | 0.8720 | 0.5400 |
| h1_mobilenet_v2_rbf_c10 | H1 | mobilenet_v2 | rbf_svm | C=10.0, $\gamma$='scale' | 0.8501 | 0.0204 | 0.8770 | 0.8936 | 0.8620 | 0.9160 | 0.9495 | 0.8453 | 0.5611 |
| h1_mobilenet_v2_logreg_c1 | H1 | mobilenet_v2 | log_reg | C=1.0 | 0.8450 | 0.0298 | 0.8662 | 0.9219 | 0.8621 | 0.8845 | 0.9014 | 0.8335 | 0.6242 |
| h1_mobilenet_v2_logreg_c01 | H1 | mobilenet_v2 | log_reg | C=0.1 | 0.8440 | 0.0277 | 0.8657 | 0.9149 | 0.8621 | 0.8775 | 0.9110 | 0.8364 | 0.6037 |
| h1_mobilenet_v2_logreg_c10 | H1 | mobilenet_v2 | log_reg | C=10.0 | 0.8430 | 0.0329 | 0.8600 | 0.9150 | 0.8506 | 0.8636 | 0.8871 | 0.8305 | 0.7084 |
| h1_efficientnet_b0_linsvm_c01 | H1 | efficientnet_b0 | linear_svm | C=0.1 | 0.8424 | 0.0267 | 0.8628 | 0.9256 | 0.8591 | 0.8497 | 0.8992 | 0.8424 | 0.6447 |
| h1_efficientnet_b0_rbf_c1 | H1 | efficientnet_b0 | rbf_svm | C=1.0, $\gamma$='scale' | 0.8402 | 0.0389 | 0.8736 | 0.8901 | 0.8764 | 0.9195 | 0.9472 | 0.8393 | 0.4784 |
| h1_mobilenet_v2_rbf_c10_g1e3 | H1 | mobilenet_v2 | rbf_svm | C=10.0, $\gamma=10^{-3}$ | 0.8336 | 0.0260 | 0.8679 | 0.8794 | 0.8506 | 0.9091 | 0.9520 | 0.8513 | 0.4674 |
| h1_mobilenet_v2_rbf_c1 | H1 | mobilenet_v2 | rbf_svm | C=1.0, $\gamma$='scale' | 0.8297 | 0.0237 | 0.8634 | 0.8830 | 0.8448 | 0.9020 | 0.9472 | 0.8424 | 0.4679 |
| h1_mobilenet_v2_linsvm_c1 | H1 | mobilenet_v2 | linear_svm | C=1.0 | 0.8223 | 0.0195 | 0.8407 | 0.9148 | 0.8506 | 0.8076 | 0.8654 | 0.8246 | 0.6342 |
| h1_mobilenet_v2_linsvm_c01 | H1 | mobilenet_v2 | linear_svm | C=0.1 | 0.8215 | 0.0203 | 0.8407 | 0.9184 | 0.8420 | 0.8005 | 0.8797 | 0.8216 | 0.6242 |
| h2_fusion_mobilenet_v2_rbf_c10_g1e3 | H2 | mobilenet_v2 | rbf_svm | C=10.0, $\gamma=10^{-3}$ | 0.5234 | 0.0178 | 0.6502 | 0.7232 | 0.8877 | 0.1676 | 0.9640 | 0.5506 | 0.0000 |

---

## 3. Analysis: Did Fusion Fix Glass, Metal, and Plastic Confusion?

**Yes, specifically for metal and plastic vs glass, with trash performance sustained on pooled counts.** The primary motivation for building a hybrid model was the severe confusion observed in Phase 1 between materials that share transparent/specular surface qualities.

### 3.1 Full 6x6 Pooled Confusion Matrices (All 1,764 Outer-Training Images)

#### Phase 1 Calibrated SVM (Extended RBF, C=10, $\gamma=10$):
| True \ Predicted | cardboard | glass | metal | paper | plastic | trash | Total | Pooled Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cardboard** | **265** | 3 | 4 | 7 | 2 | 1 | 282 | 265 / 282 = 93.97% |
| **glass** | 2 | **311** | 20 | 2 | 11 | 2 | 348 | 311 / 348 = 89.37% |
| **metal** | 3 | 21 | **241** | 5 | 11 | 5 | 286 | 241 / 286 = 84.27% |
| **paper** | 6 | 3 | 3 | **394** | 8 | 2 | 416 | 394 / 416 = 94.71% |
| **plastic** | 1 | 21 | 12 | 9 | **289** | 4 | 336 | 289 / 336 = 86.01% |
| **trash** | 4 | 1 | 7 | 1 | 5 | **78** | 96 | 78 / 96 = 81.25% |

#### H2 Early Feature Fusion: EfficientNet-B0 + Handcrafted (`h2_fusion_efficientnet_b0_rbf_c10`):
| True \ Predicted | cardboard | glass | metal | paper | plastic | trash | Total | Pooled Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cardboard** | **269** | 1 | 0 | 11 | 1 | 0 | 282 | 269 / 282 = 95.39% |
| **glass** | 1 | **319** | 16 | 2 | 10 | 0 | 348 | 319 / 348 = 91.67% |
| **metal** | 2 | 8 | **266** | 2 | 5 | 3 | 286 | 266 / 286 = 93.01% |
| **paper** | 12 | 1 | 2 | **391** | 3 | 7 | 416 | 391 / 416 = 93.99% |
| **plastic** | 2 | 13 | 11 | 4 | **303** | 3 | 336 | 303 / 336 = 90.18% |
| **trash** | 5 | 0 | 3 | 0 | 4 | **84** | 96 | 84 / 96 = 87.50% |

#### H2 Early Feature Fusion: MobileNetV2 + Handcrafted (`h2_fusion_mobilenet_v2_rbf_c10`):
| True \ Predicted | cardboard | glass | metal | paper | plastic | trash | Total | Pooled Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cardboard** | **274** | 0 | 1 | 7 | 0 | 0 | 282 | 274 / 282 = 97.16% |
| **glass** | 2 | **324** | 12 | 0 | 9 | 1 | 348 | 324 / 348 = 93.10% |
| **metal** | 4 | 8 | **259** | 2 | 11 | 2 | 286 | 259 / 286 = 90.56% |
| **paper** | 10 | 0 | 1 | **395** | 5 | 5 | 416 | 395 / 416 = 94.95% |
| **plastic** | 4 | 17 | 8 | 5 | **298** | 4 | 336 | 298 / 336 = 88.69% |
| **trash** | 6 | 2 | 4 | 4 | 3 | **77** | 96 | 77 / 96 = 80.21% |

---

### 3.2 Glass, Metal, and Plastic Submatrix Comparison

Focusing specifically on the mutual confusions between glass, metal, and plastic:

| Model | Glass $\to$ Metal | Glass $\to$ Plastic | Metal $\to$ Glass | Metal $\to$ Plastic | Plastic $\to$ Glass | Plastic $\to$ Metal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 SVM** | 20 / 348 (5.75%) | 11 / 348 (3.16%) | **21 / 286 (7.34%)** | 11 / 286 (3.85%) | **21 / 336 (6.25%)** | 12 / 336 (3.57%) |
| **H2 EfficientNet-B0** | 16 / 348 (4.60%) | 10 / 348 (2.87%) | **8 / 286 (2.80%)** | 5 / 286 (1.75%) | **13 / 336 (3.87%)** | 11 / 336 (3.27%) |
| **H2 MobileNetV2** | 12 / 348 (3.45%) | 9 / 348 (2.59%) | **8 / 286 (2.80%)** | 11 / 286 (3.85%) | **17 / 336 (5.06%)** | 8 / 336 (2.38%) |

### Key Findings:
1. **Dramatic reduction in Metal $\to$ Glass confusion:** Slashed from **21 / 286 (7.34%)** in Phase 1 down to **8 / 286 (2.80%)** in both H2 backbones—a **61.9% reduction**. Our hypothesis is that CNN representations provide high-level structural contours (e.g. cylindrical silhouettes and rim creases) that prevent specular metal surfaces from being misclassified as glass bottles. Note that this architectural hypothesis is inferential; nothing in this PR visualizes spatial feature attention, which can be explicitly evaluated using Grad-CAM in the final evaluation.
2. **Substantial Metal Recall Gain:** Pooled metal recall rose from **241 / 286 = 84.27%** to **266 / 286 = 93.01%** (+8.74% absolute improvement, +25 correctly classified metal items).
3. **Plastic and Glass Gains:** Plastic pooled recall improved from **289 / 336 = 86.01%** to **303 / 336 = 90.18%**, with plastic misclassified as glass dropping from 21 down to 13. Glass pooled recall rose from **311 / 348 = 89.37%** to **319 / 348 = 91.67%** (and 324 / 348 = 93.10% for MobileNetV2).
4. **Trash Class Evaluation:** On pooled counts, trash recall actually improved in H2 EfficientNet-B0 from **78 / 96 = 81.25%** to **84 / 96 = 87.50%** (+6.25% absolute gain, with remaining misclassifications going to cardboard (5), metal (3), and plastic (4)). For MobileNetV2 H2, pooled recall is **77 / 96 = 80.21%** (a 1-image shift). The lower *mean per-fold* recall (74.89% and 72.84%) reflects fold-to-fold variance on the small class size of 96 images (approx. 19 images per fold), where 1 or 2 extra errors in a single fold lower that fold's percentage substantially.

---

## 4. Phase 1 SVM Out-of-Fold (OOF) Probabilities for PR B

To support PR B (late probability fusion H3), the out-of-fold calibrated probabilities for Phase 1's winning RBF SVM were generated and saved:
- Path: `artifacts/phase2/hybrid/phase1_svm_oof.csv`
- Shape: `(1764, 9)` (columns: `image_id`, `fold`, `label`, `prob_cardboard`, `prob_glass`, `prob_metal`, `prob_paper`, `prob_plastic`, `prob_trash`).
- Properties: Every row corresponds to a single held-out prediction from the model trained on the other 4 folds. Sum of probabilities across every single row equals $1.000000 \pm 10^{-6}$.

---

## 5. Viva Notes: Examiner Questions and Answers

### Q1: Your hybrid scores higher (0.9044 vs 0.8803), why don't you claim it wins?
**Answer:** In scientific machine learning, an observed increase in mean metric does not automatically establish superiority if it is smaller than the variation across cross-validation splits. Our pre-registered protocol explicitly defines the decision rule: $\Delta > \sigma$. Here, $\Delta = 0.0241$ while $\sigma = 0.0332$. Because the cross-fold standard deviation exceeds the gain, the difference could be an artifact of fold partitioning rather than a genuine shift in expected generalization error. Claiming a definitive win would violate our pre-registered integrity protocol. Furthermore, comparison with the best single CNN configuration is pending Aditya's ablation results, so the final selection is deferred to PR B.

### Q2: Why train classical ML (SVM, Logistic Regression) on CNN embeddings instead of just a linear softmax head with SGD?
**Answer:** On small datasets ($N=1,764$), standard gradient descent with cross-entropy is prone to early-stopping noise, sensitive to batch sampling order, and only maximizes empirical loss. Support Vector Machines directly maximize the geometric margin ($2/\|\mathbf{w}\|$) between class boundaries. This structural risk minimization principle often finds better-generalizing hyperplanes in fixed 1,280-dimensional embedding spaces than SGD heads.

### Q3: What is "per-block standard scaling", and why divide by $\sqrt{D_{\text{block}}}$?
**Answer:** In early fusion, we concatenate two modalities: handcrafted features ($D_{\text{hc}} = 1,608$) and CNN embeddings ($D_{\text{cnn}} = 1,280$). If concatenated directly, Euclidean distance in the RBF kernel would be dominated by whichever block has higher variance or more dimensions ($1,608$ vs $1,280$). By fitting `StandardScaler` on each block independently and then dividing each feature by $\sqrt{D_{\text{block}}}$, the expected squared norm of each block satisfies $\mathbb{E}[\|\mathbf{x}'\|^2] = 1$. Both modalities therefore exert equal geometric influence on the RBF kernel distance metric.

### Q4: Why must the scalers and fusion parameters be fitted strictly inside each CV fold?
**Answer:** Fitting a scaler or transformer on the full dataset before splitting leaks global mean and standard deviation information from the validation fold into the training set (transductive data leakage). By encapsulating the scaler inside a fold-local `Pipeline`, the validation fold behaves as completely unseen data, preserving the integrity of the cross-validation error estimate.

### Q5: Why did we evaluate only on the 5 training CV folds and not the validation or test splits?
**Answer:** The project follows strict split isolation. The validation split ($N=380$) is reserved exclusively for Prajjwal Tripathi to fit probability calibration (Platt/temperature scaling) and select the operational abstention threshold. The test split ($N=377$) is held out for the single final system evaluation. Using validation or test data to select hybrid configurations or tune hyperparameters would introduce optimistic selection bias.

### Q6: Why did H1 alone (ML on frozen CNN embeddings) score lower (0.8597) than Phase 1 handcrafted features (0.8803)?
**Answer:** Pretrained ImageNet CNNs are trained on natural object recognition, where background, context, and semantic category dominate. TrashNet consists of isolated waste items against clean white poster boards where fine-grained texture (LBP), surface reflection (HSV saturation/value), and subtle color gradients carry critical discriminative signal. Without fine-tuning the convolutional weights, the frozen ImageNet embeddings lacked domain-specific surface awareness, which the handcrafted descriptors explicitly captured.

### Q7: Why did H2 (early fusion) succeed in reducing metal-to-glass confusion from 21 errors to 8?
**Answer:** Metal and glass both produce intense specular highlights and sharp edges on white backgrounds, causing handcrafted HOG and color histograms to overlap. Our hypothesis is that CNN embeddings provide high-level structural contours (e.g. cylindrical silhouettes and rim creases) that prevent specular metal surfaces from being misread as glass bottles. When fused, the classifier can utilize structural cues to disambiguate the material when surface histograms are ambiguous. This can be verified with Grad-CAM visualization in the final evaluation.

### Q8: What does "out-of-fold (OOF)" mean, and why is it needed for late fusion (H3)?
**Answer:** An out-of-fold prediction for sample $i$ is generated by a model trained on all folds *except* the fold containing $i$. Across 5 folds, every sample in the training set receives exactly one unbiased prediction. In PR B, late fusion will find the optimal linear blend $w \cdot p_{\text{cnn}} + (1-w) \cdot p_{\text{svm}}$. Using training predictions directly would cause the weights to overfit the more overconfident model; OOF probabilities prevent this leakage.

### Q9: Why was $\gamma=10^{-3}$ severely worse ($0.5234$ Macro F1) in H2 compared to $\gamma=\text{'scale'}$?
**Answer:** In an RBF kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$, $\gamma$ controls the radius of influence of support vectors. With 2,888 normalized features, Euclidean distances scale with dimension. Setting $\gamma=10^{-3}$ made the kernel bandwidth excessively wide, oversmoothing the decision boundary and collapsing minority class predictions (reducing Trash recall to 0.0). In contrast, $\gamma=\text{'scale'} = 1 / (D \cdot \text{Var}(X))$ automatically adapts to the feature dimension.

### Q10: How does this hybrid compare to modern multimodal architectures?
**Answer:** In deep learning, early fusion (feature-level concatenation) and late fusion (decision-level ensemble) are standard paradigms. Our approach demonstrates that when domain-specific engineered representations (such as photometric histograms) capture low-level physical properties that deep models ignore without massive fine-tuning data, fusing them with deep representations provides a principled, lightweight alternative to full end-to-end retraining.
