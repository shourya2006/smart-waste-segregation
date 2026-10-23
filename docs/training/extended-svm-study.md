# Extended colour/texture SVM study

This opt-in experiment follows the completed original-feature study (best macro
F1 0.8042). The requested 0.86+ macro F1 is an aspiration, not a guaranteed result.
The earlier CV results informed this follow-up; all subsequent selection scores
are exploratory and require a frozen final test evaluation later.

## Declared representation and search

Before fitting, `extended_study.py` writes a plan with source/environment, the
original training fingerprint, fixed-fold checksum, descriptor settings and grid.
The experiment reads only the 1,764 audited outer-training images, verifies their
SHA-256 values, and preserves the original five group-aware fold assignments.
A separate derived-feature fingerprint records the new matrix; it does not claim
the old features and new features are identical. No outer validation/test image
is extracted by this experiment. The original 1,806-feature cache is unchanged.

`extended_features.py` calls the shared preprocessing implementation with an
explicit 256×256 canvas. It computes six regions: full image, a fixed foreground
mask (HSV saturation >0.12 OR value <0.92), and four quadrants. The foreground
mask is a heuristic; pale objects may be omitted, so the full-image region is
retained in every family. Spatial features may depend on object position and
background and do not establish generalization to everyday photographs.

Each region contains:

- 288 colour bins: HSV 32/16/16; RGB 16/channel; Lab 16/channel with fixed ranges;
  and joint hue/saturation 16×8.
- 172 texture bins: rotation-invariant uniform LBP at (P,R)=(8,1),(16,2),(24,3),
  plus rotation-sensitive `nri_uniform` LBP at (8,1),(8,2), 59 bins each.

Each component histogram has unit mass (zero if empty), then colour and texture
families each have unit mass per region. LBP excludes the outer three-pixel border.
The complete input has 2,760 nonnegative float32 entries. Pipelines select columns
internally and apply square-root/L2 normalization independently to each image.
No global scaler, feature selection or codebook is learned outside a fold.

Four ablations: global colour (576), global colour+texture (920), global colour
with spatial texture (1,608), and full spatial colour+texture (2,760). Here
“global” includes both the full-image and foreground regions.
Each uses an RBF SVM with C=10/30/100, gamma=1/5/10 and class weight none/balanced:
18 candidates per family, 72 total, 360 CV fits and four training refits. The
winner is selected by mean-fold macro F1, with family-name tie-breaking, followed
by five OOF diagnostic fits. The plan stops after these candidates.

## Reproduction and handoff

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m waste_sorting.extended_study \
  --output artifacts/shourya-extended-svm-study --n-jobs 3
```

The output directory must be new. Models and extracted feature arrays are ignored
local artifacts. The winning model expects `extract_extended_features(image)`;
it cannot receive the original 1,806-feature vector. Prajjwal's eventual image
prediction/calibration code must select the extractor using the saved experiment
metadata. Calibration and test evaluation are not part of this study.

Reference implementation documentation:
[scikit-image LBP API](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern)
and [LBP texture example](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html).

## Verified outcome

All 72 candidates completed. The winner is `global_color_spatial_texture`: mean fold macro F1 **0.880263**, fold standard deviation **0.021545**, pooled OOF accuracy **0.890023**, pooled macro F1 **0.881185**. Thus the requested 0.86 mean-CV target was exceeded; untouched-test performance remains unknown.

Selected settings: RBF SVM, C=10, gamma=10, no class weighting; 1,608 internally selected columns from the 2,760-dimensional extractor output.

All four family winners:

| Family | Mean CV macro F1 | Fold std |
| --- | ---: | ---: |
| global_color | 0.833778 | 0.022842 |
| global_color_texture | 0.869583 | 0.018834 |
| global_color_spatial_texture | 0.880263 | 0.021545 |
| spatial_color_texture | 0.869644 | 0.020731 |

The winning fold scores are 0.878880, 0.858753, 0.907015, 0.902138, 0.854529. Two folds remain below 0.86; exceeding the mean target does not establish a lower bound across folds or future data.

Trash remains the weakest class by recall (0.78125, 75/96 correct). Metal-to-glass confusion (24 images) is the largest off-diagonal count. Class support totals 1,764. Report results for every class; no examples/classes were excluded to raise the metric.

Verification: all 82 tests pass; lint and dependency checks pass. Independently checked all 72 candidate means/stds, four saved pipelines, selected parameters, full input/selected dimensions, raw-image/cache inference parity, recorded source/feature/fold checksums, and OOF identities/fold assignments/metrics.

[Full results and plots](extended-svm-results/README.md). The winning local artifact is `artifacts/shourya-extended-svm-study/global_color_spatial_texture/estimator.joblib`. It is fitted on outer training only and remains uncalibrated.

```python
import joblib
from PIL import Image
from waste_sorting.extended_features import extract_extended_features

model = joblib.load("artifacts/shourya-extended-svm-study/global_color_spatial_texture/estimator.joblib")
with Image.open("your-image.jpg") as image:
    vector = extract_extended_features(image)
class_index = int(model.predict(vector[None, :])[0])
# Resolve the index through summary.json -> diagnostics -> class_order.
# This is a class prediction; calibrated confidence is not implemented here.
```
