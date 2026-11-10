# Phase 2 Backbones and Pretrained Embeddings

**Owner:** Aditya Bhardwaj  
**Reviewer:** Om Yadav  
**Code:** [`src/waste_sorting/backbones.py`](../../src/waste_sorting/backbones.py), [`src/waste_sorting/embeddings.py`](../../src/waste_sorting/embeddings.py)  
**Configuration:** [`configs/phase2.toml`](../../configs/phase2.toml)  
**Tests:** [`tests/test_backbones.py`](../../tests/test_backbones.py), [`tests/test_embeddings.py`](../../tests/test_embeddings.py)

---

## 1. Interface for Teammates

### 1.1 Summary Table

| Function / Tool | Module | Arguments | Returns / Output |
| :--- | :--- | :--- | :--- |
| `load_backbone(name, pretrained=True)` | `waste_sorting.backbones` | `name: str` (`"mobilenet_v2"` or `"efficientnet_b0"`), `pretrained: bool` | `tuple[nn.Module, int]`: `(feature_extractor, 1280)` mapping `[B, 3, H, W] -> [B, 1280]` |
| `freeze(model)` | `waste_sorting.backbones` | `model: nn.Module` | `None`: Sets `requires_grad = False` for all model parameters |
| `keep_frozen_bn_eval(model)` | `waste_sorting.backbones` | `model: nn.Module` | `None`: Sets any BatchNorm module whose parameters are fully frozen into `eval` mode. **Must be called after every `model.train()`** |
| `unfreeze_last_blocks(model, n)` | `waste_sorting.backbones` | `model: nn.Module`, `n: int` | `None`: Sets `requires_grad = True` for trailing `n` feature blocks/stages |
| `count_trainable_parameters(model)` | `waste_sorting.backbones` | `model: nn.Module` | `int`: Total number of parameters with `requires_grad=True` |
| `extract_embeddings(backbone_name, config, root=".", split="train", ...)` | `waste_sorting.embeddings` | `backbone_name: str`, `config: dict`, `root: Path | str`, `split: str` (`"train"`, `"validation"`, `"test"`), `force: bool = False` | `Path`: Path to saved `.npz` embedding cache |
| `load_embeddings(backbone_name, split, root=".", config=None)` | `waste_sorting.embeddings` | `backbone_name: str`, `split: str`, `root: Path | str`, `config: dict | None` | `dict` containing `embeddings` (`[N, 1280]` float32), `labels` (`[N]` int64), `image_ids` (`[N]` str), and `metadata` (`dict`). Refuses stale caches! |
| `waste-dl-embeddings` | CLI (`waste_sorting.embeddings`) | `--backbone {mobilenet_v2,efficientnet_b0}`, `--splits train validation test`, `[--force]` | Extracts and caches embeddings to `artifacts/phase2/embeddings/<backbone>/<split>.npz` |

---

### 1.2 Usage Examples

#### Shourya's Training Loop (Frozen Head Training & Partial Fine-Tuning)
```python
import torch
from waste_sorting.backbones import freeze, keep_frozen_bn_eval, load_backbone, unfreeze_last_blocks

# 1. Load backbone with pretrained weights and freeze parameters
backbone, embedding_dim = load_backbone("mobilenet_v2", pretrained=True)
freeze(backbone)

# 2. Attach a classification head (e.g. Linear / MLP for 6 classes)
classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(embedding_dim, 6))

# 3. For fine-tuning stage: unfreeze last n blocks (e.g. n=2)
unfreeze_last_blocks(backbone, n=2)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, list(backbone.parameters()) + list(classifier.parameters())),
    lr=1e-4,
)

# IMPORTANT: Whenever switching to train mode, call keep_frozen_bn_eval right after model.train():
backbone.train()
classifier.train()
keep_frozen_bn_eval(backbone)  # Prevents frozen BatchNorm running statistics from corrupting!
```

#### Daksh's Hybrid Models (SVM / Classical Classifiers on CNN Embeddings)
```python
from waste_sorting.embeddings import load_embeddings

# Load verified, frozen CNN feature embeddings directly from disk (refuses stale cache)
train_data = load_embeddings("mobilenet_v2", split="train")
X_train, y_train = train_data["embeddings"], train_data["labels"]

val_data = load_embeddings("mobilenet_v2", split="validation")
X_val, y_val = val_data["embeddings"], val_data["labels"]
# Train classical classifiers (SVM, RF) or concatenate with Phase 1 HSV/HOG/LBP
```

---

## 2. Architecture Theory for the Viva

### 2.1 MobileNetV2 (Sandler et al., 2018)
- **Depthwise-Separable Convolutions:** Standard convolutions jointly map cross-channel correlations and spatial correlations at a computational cost of $D_K \times D_K \times M \times N \times D_F \times D_F$. Depthwise-separable convolutions factorize this operation into:
  1. A **depthwise convolution** applying a single convolutional filter per input channel ($D_K \times D_K \times M \times D_F \times D_F$).
  2. A **pointwise convolution** ($1 \times 1$ conv) computing linear combinations across channels ($1 \times 1 \times M \times N \times D_F \times D_F$).
  This reduces computational cost by a factor of roughly $\frac{1}{N} + \frac{1}{D_K^2}$, achieving an $\approx 8$ to $9\times$ reduction for $3 \times 3$ kernels with minimal accuracy drop.
- **Inverted Residuals:** Classical residual blocks (e.g. ResNet) employ a *bottleneck* structure that compresses channel dimension ($d \to d/4$), applies standard convs, and expands back ($d/4 \to d$). Inverted residuals reverse this: they take a low-dimensional compressed representation, expand it via $1 \times 1$ conv to a higher-dimensional space ($tk$, where expansion factor $t=6$), apply depthwise $3 \times 3$ convolutions, and project back to a low-dimensional representation ($k$). Residual shortcuts connect the low-dimensional bottlenecks directly, which substantially decreases memory consumption during backpropagation.
- **Linear Bottlenecks:** Non-linear activation functions (like ReLU) zero out negative activations. When applied to low-dimensional manifold representations, ReLU destroys significant information (collapsing manifolds). MobileNetV2 therefore intentionally omits non-linear activations after the final projection $1 \times 1$ convolution, maintaining a strictly linear bottleneck.

### 2.2 EfficientNet-B0 (Tan & Le, 2019)
- **MBConv & Squeeze-and-Excitation (SE):** EfficientNet-B0 uses mobile inverted bottleneck convolution (MBConv) blocks enhanced with Squeeze-and-Excitation mechanisms. SE adaptively recalibrates channel-wise feature responses:
  1. **Squeeze:** Global average pooling squeezes spatial dimensions ($H \times W$) into a $1 \times 1 \times C$ channel descriptor.
  2. **Excitation:** A two-layer bottleneck MLP (reduction ratio $r=4$ or $r=24$) with a Sigmoid gating activation captures channel dependencies and scales the feature maps multiplicatively.
- **Compound Scaling:** Prior architectures scaled single dimensions arbitrarily: network depth $d$ (ResNet-50 to ResNet-152), network width $w$ (WideResNet), or image resolution $r$. Tan & Le established that balancing depth, width, and resolution yields superior accuracy under fixed resource budgets:
  $$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
  subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha \ge 1, \beta \ge 1, \gamma \ge 1$, where $\phi$ is the global compound scaling coefficient and $\alpha, \beta, \gamma$ are determined by grid search on the baseline architecture (B0). EfficientNet-B0 is the foundational base model discovered via neural architecture search (NAS).

### 2.3 Model Complexity: Parameters and FLOPs
| Metric | MobileNetV2 | EfficientNet-B0 |
| :--- | :--- | :--- |
| **Total Parameters (with ImageNet head)** | $\approx 3.50\text{ M}$ | $\approx 5.29\text{ M}$ |
| **Backbone Extractor Parameters** | $2,223,872$ | $4,007,548$ |
| **FLOPs ($224 \times 224$ input)** | $\approx 300\text{ MFLOPs}$ | $\approx 390\text{ MFLOPs}$ |
| **Embedding Dimension** | $1280$ | $1280$ |
| **Default Weights Pinned** | `IMAGENET1K_V2` | `IMAGENET1K_V1` |

### 2.4 Structural Feature Blocks & Trainable Parameter Counts for Fine-Tuning

When calling `unfreeze_last_blocks(model, n)`, trailing structural modules in `model.features[-n:]` are set to `requires_grad=True`:

- **MobileNetV2 (`model.features` contains 19 elements):**
  - Index 0: `Conv2dNormActivation` (initial $3 \times 3$ conv)
  - Index 1..17: 17 `InvertedResidual` blocks across 7 stages
  - Index 18: `Conv2dNormActivation` (final $1 \times 1$ conv expanding to 1,280 channels)
  - `n=1`: Unfreezes index 18 (final $1 \times 1$ conv)
  - `n=2`: Unfreezes indices 17 and 18 (last `InvertedResidual` block + final conv)
  - `n=3`: Unfreezes indices 16, 17, and 18 (last two `InvertedResidual` blocks + final conv)

- **EfficientNet-B0 (`model.features` contains 9 stages):**
  - Stage 0: `Conv2dNormActivation` (stem $3 \times 3$ conv)
  - Stage 1..7: 7 sequential stages of `MBConv` blocks (containing 1, 2, 2, 3, 3, 4, 1 MBConv blocks respectively)
  - Stage 8: `Conv2dNormActivation` (final $1 \times 1$ conv expanding to 1,280 channels)
  - `n=1`: Unfreezes stage 8 (final $1 \times 1$ conv)
  - `n=2`: Unfreezes stages 7 and 8 (last `MBConv` stage + final conv)
  - `n=3`: Unfreezes stages 6, 7, and 8 (last two `MBConv` stages + final conv)

#### Trainable Parameter Counts ($n = 1, 2, 3$)
*(Computed via `count_trainable_parameters(model)` after `freeze(model)` and `unfreeze_last_blocks(model, n)`):*

| Backbone | Mode / $n$ | Unfrozen Feature Modules | Trainable Backbone Params | % of Total Backbone |
| :--- | :--- | :--- | ---: | ---: |
| **MobileNetV2** | $n=0$ (Frozen) | None (head only) | 0 | 0.0% |
| | $n=1$ | `features[18]` (final $1 \times 1$ conv) | 412,160 | 18.5% |
| | $n=2$ | `features[17:19]` (last InvertedResidual + final conv) | 886,080 | 39.8% |
| | $n=3$ | `features[16:19]` (last 2 InvertedResiduals + final conv) | 1,206,080 | 54.2% |
| **EfficientNet-B0** | $n=0$ (Frozen) | None (head only) | 0 | 0.0% |
| | $n=1$ | `features[8]` (final $1 \times 1$ conv) | 412,160 | 10.3% |
| | $n=2$ | `features[7:9]` (stage 7 MBConv + final conv) | 1,129,392 | 28.2% |
| | $n=3$ | `features[6:9]` (stages 6 & 7 MBConvs + final conv) | 3,155,740 | 78.7% |

### 2.5 Why ImageNet Transfer Learning Works with 1,764 Training Images
1. **Sample Efficiency:** Training deep CNNs from scratch on 1,764 images leads to catastrophic overfitting: the number of free parameters ($>2-4\text{ M}$) dwarfs the dataset size by three orders of magnitude.
2. **Feature Transferability (Yosinski et al., 2014):** Early layers in deep networks learn general, Gabor-like edge, blob, and texture primitives that are universal across visual domains. Mid-level layers capture shape fragments, surface boundaries, and contours. Only late layers encode task-specific semantic classes.
3. **Domain Alignment:** TrashNet items (glass bottles, soda cans, cardboard boxes, paper sheets) share identical low-level and mid-level geometric primitives with ImageNet categories (containers, paper products, glassware, household goods).
4. **Frozen Extraction vs. Fine-Tuning:**
   - *Frozen feature extraction* treats the backbone as a fixed non-linear feature extractor, optimizing only the lightweight classification head ($\approx 7,680$ parameters for 6 classes). This completely eliminates the risk of gradient descent corrupting pretrained filters ("catastrophic forgetting").
   - *Partial fine-tuning* keeps early layers frozen while unfreezing top structural blocks with a reduced learning rate ($10^{-4}$), allowing high-level features to specialize to waste-specific lighting, specular reflections, and deformation patterns without destroying low-level feature generalizability.

### 2.5 Deliberate Preprocessing Choice: Letterbox vs. Center-Crop
Torchvision's default weights transforms perform `Resize` followed by `CenterCrop(224)`. For waste sorting, center-cropping regularly clips key diagnostic object boundaries (e.g., bottle necks, can pull-tabs, box edges). We deliberately route all images through Om's Phase 1 aspect-preserving letterbox with white padding, followed by ImageNet mean/standard deviation normalization. This guarantees that objects are neither cropped nor distorted.

---

## 3. Viva Notes: Examiner Questions & Model Answers

1. **Q: Why did you choose MobileNetV2 and EfficientNet-B0 rather than larger networks like ResNet-50 or ViT?**  
   *A:* Waste segregation systems are intended for edge and embedded deployment (e.g., smart recycling bins equipped with Jetson Nano or Raspberry Pi). MobileNetV2 ($\approx 3.5\text{M}$ params, $300\text{ MFLOPs}$) and EfficientNet-B0 ($\approx 5.3\text{M}$ params, $390\text{ MFLOPs}$) provide state-of-the-art accuracy-to-latency tradeoffs while running efficiently on CPU-only hardware. Furthermore, with only 1,764 training images, smaller backbones have significantly lower capacity to overfit compared to heavy models.

2. **Q: How does a depthwise-separable convolution save computation mathematically?**  
   *A:* A standard conv with kernel size $D_K \times D_K$, input channels $M$, output channels $N$, and output spatial size $D_F \times D_F$ requires $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$ operations. A depthwise-separable convolution splits this into a depthwise step ($D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$) and a pointwise $1 \times 1$ step ($M \cdot N \cdot D_F \cdot D_F$). The computational ratio is $\frac{D_K^2 \cdot M + M \cdot N}{D_K^2 \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}$. For a $3 \times 3$ filter, this represents an approximate $8\times$ to $9\times$ reduction in arithmetic operations.

3. **Q: What is the purpose of the linear bottleneck in MobileNetV2? Why not use ReLU?**  
   *A:* Deep neural networks operate on manifolds of interest embedded in high-dimensional spaces. When an inverted residual block projects features back to a low-dimensional bottleneck ($k \ll tk$), applying non-linearities like ReLU zeroes out all negative coordinates. In low-dimensional spaces, this collapses volume and permanently destroys visual information. Retaining a linear transformation prevents manifold collapse.

4. **Q: How does EfficientNet's Squeeze-and-Excitation block work?**  
   *A:* It models channel interdependencies. The "Squeeze" step uses global average pooling across $H \times W$ to generate a channel-wise statistic vector of length $C$. The "Excitation" step passes this vector through a two-layer bottleneck MLP with a reduction ratio to capture non-linear channel relationships, ending in a Sigmoid activation. The resulting scalar weights scale each feature map multiplicatively, amplifying informative channels and suppressing noise.

5. **Q: What is compound scaling and why is it superior to scaling depth alone?**  
   *A:* Scaling only depth (like ResNet-101 to ResNet-152) suffers diminishing returns due to vanishing gradients and receptive field saturation. Scaling only width results in shallow networks that miss complex high-level abstractions. Scaling only resolution increases pixel count without adequate receptive field or capacity. Compound scaling balances all three dimensions using fixed scaling coefficients ($\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$), ensuring receptive fields match larger input resolutions while network depth and width expand proportionally.

6. **Q: Why do both MobileNetV2 and EfficientNet-B0 output 1280-dimensional vectors?**  
   *A:* Both architectures terminate their convolutional stages in an expansion layer followed by adaptive global average pooling ($1 \times 1$). In MobileNetV2, block 18 expands channels to 1,280 prior to average pooling. In EfficientNet-B0, stage 8 also projects channels to 1,280. Stripping torchvision's linear classification head exposes this exact 1,280-dimensional feature representation.

7. **Q: Why extract embeddings with the eval transform rather than augmented training transforms?**  
   *A:* Feature embeddings must represent the deterministic, canonical representation of each sample. Caching stochastic augmented features would introduce random variance into downstream hybrid models and invalidate cache provenance. Data augmentation belongs exclusively inside the stochastic gradient descent loop during end-to-end training.

8. **Q: How does your embedding cache prevent data leakage and stale experiment runs?**  
   *A:* Each `.npz` archive embeds a cryptographically signed metadata JSON containing the SHA-256 hash of the audited manifest, the configuration hash of preprocessing/DL settings, the source code fingerprint of the extraction pipeline, the pinned torchvision weights version, and the git commit hash. If any parameter, split assignment, or code change occurs, `load_embeddings()` refuses the cache with an informative error rather than silently serving outdated data.

9. **Q: Did you evaluate or inspect the test set embeddings?**  
   *A:* Absolutely not. The test set embeddings were extracted and cached to disk alongside train and validation splits to enable seamless final inference, but no evaluation, scoring, clustering, or inspection was performed on test data. All architectural selections and hyperparameter choices are pre-registered and determined solely via 5-fold cross-validation on the training set.

---

## 4. Report Contribution: IEEE Architecture Paragraph

```latex
To transfer deep representations to waste categorization under severe data scarcity (1,764 training images), we leverage two lightweight convolutional backbones pretrained on ImageNet-1K: MobileNetV2~\cite{sandler2018mobilenetv2} and EfficientNet-B0~\cite{tan2019efficientnet}. MobileNetV2 factorizes standard spatial convolutions into depthwise-separable convolutions and inverted residual blocks with linear bottlenecks, preserving manifold geometry in low-dimensional projections while constraining computational complexity to $\sim$300 MFLOPs. EfficientNet-B0 optimizes mobile inverted convolutions (MBConv) via compound scaling, balancing depth, width, and input resolution alongside channel-wise squeeze-and-excitation attention mechanisms~\cite{tan2019efficientnet}. In accordance with empirical findings on feature transferability~\cite{yosinski2014transferable}, general low-level edge and textural filters are preserved by freezing early backbone stages, mapping each input image through adaptive average pooling to a dense 1,280-dimensional embedding vector. To prevent diagnostic boundary clipping common to standard center-cropping, all inputs undergo aspect-ratio-preserving letterboxing with ImageNet standardization prior to feature extraction.
```

---

## 5. Bibliography Entries

Appended to [`report/references.bib`](../../report/references.bib):
- `sandler2018mobilenetv2` (MobileNetV2, CVPR 2018)
- `tan2019efficientnet` (EfficientNet, ICML 2019)
- `yosinski2014transferable` (Transferable Features, NeurIPS 2014)

