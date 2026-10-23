# Aditya Bharadwaj Feature Extraction and Caching Handoff

Prepared on 23 October 2026 with the coding assistant for Aditya's assigned component.
This document details the implementation, mathematical design choices, normalization policies,
reproducibility guarantees, teammate usage instructions, and verified results.

## Implemented Interfaces

### 1. Unified Feature Extraction
```python
from waste_sorting.features import extract_features

features = extract_features(image, config, descriptors=None)
```
- `image`: PIL Image (raw RGB, RGBA, grayscale, or with EXIF tags).
- `config`: Complete project configuration mapping (from `configs/phase1.toml`).
- `descriptors`: Optional sequence of descriptor identifiers (e.g. `["color"]`, `["hog"]`, `["lbp"]`,
  or combinations like `["color", "hog"]`). Defaults to `("color", "hog", "lbp")`.
- **Return value**: A 1D `numpy.ndarray` of type `float32` with all finite values. Under default configuration
  with all three descriptors enabled, it contains **exactly 1,806 features**.

### 2. Individual Descriptor Functions
Specialized helper extractors for feature ablations and comparisons:
- `extract_hsv_features(image, config) -> np.ndarray`: 32 features.
- `extract_hog_features(image, config) -> np.ndarray`: 1,764 features.
- `extract_lbp_features(image, config) -> np.ndarray`: 10 features.
- `get_feature_names(config, descriptors=None) -> list[str]`: 1,806 descriptive feature names.
- `get_descriptor_boundaries(config, descriptors=None) -> dict[str, tuple[int, int]]`:
  Returns slice boundaries: `{"color": (0, 32), "hog": (32, 1796), "lbp": (1796, 1806)}`.

### 3. Loading Cached Features (For Shourya, Daksh, and Prajjwal)
```python
from waste_sorting.features import load_features

data = load_features("data/processed/features.npz", root=Path.cwd())
X = data["features"]            # (2521, 1806) float32
y = data["targets"]             # (2521,) int64 class indices
splits = data["splits"]         # (2521,) strings: "train", "validation", "test"
groups = data["duplicate_groups"]  # (2521,) duplicate group identifiers
classes = data["classes"]       # ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Example: Selecting training fold data
train_mask = splits == "train"
X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
```

---

## Descriptor Design & Normalization Choices

### 1. HSV Color Histograms (32 features)
- **Grayscale/Color conversion**: Converted to RGB via shared preprocessing, then mapped to HSV using
  `skimage.color.rgb2hsv` with channels in $[0.0, 1.0]$.
- **Binning**: 16 Hue bins, 8 Saturation bins, 8 Value bins (as declared in `[features.color.bins]`).
- **Rationale**: Separate channel histograms capture dominant color and brightness profiles with only 32 dimensions.
  A joint 3D histogram ($16 \times 8 \times 8 = 1,024$ bins) would be excessively sparse for $128 \times 128$ images.
- **Normalization**: Each channel histogram is independently L1-normalized by dividing by its pixel count ($N$),
  so $\sum h = 1.0$, $\sum s = 1.0$, and $\sum v = 1.0$. This ensures the color description forms empirical probability
  distributions invariant to resolution and prevents illumination intensity from dominating chromaticity.

### 2. Histogram of Oriented Gradients (HOG) (1,764 features)
- **Grayscale conversion**: RGB converted to luminance grayscale in $[0.0, 1.0]$ using standard Rec. 709 luma weights
  ($Y = 0.2125R + 0.7154G + 0.0721B$) via `skimage.color.rgb2gray`.
- **Parameters**: 9 orientation bins in $[0, \pi)$, $16 \times 16$ pixels per cell, $2 \times 2$ cells per block.
- **Dimension calculation**: For a $128 \times 128$ input canvas:
  $$\text{Cells across} = 128 / 16 = 8, \quad \text{Blocks across} = 8 - 2 + 1 = 7$$
  $$\text{Total blocks} = 7 \times 7 = 49$$
  $$\text{Features per block} = 2 \times 2 \times 9 = 36$$
  $$\text{Total HOG features} = 49 \times 36 = 1,764$$
- **Normalization**: Standard Dalal-Triggs `L2-Hys` block normalization (L2-norm followed by clipping values at 0.2
  and renormalizing by L2-norm). This suppresses gradient burst artifacts from specular reflections (common on glass/metal)
  and illumination variations across image backgrounds.

### 3. Uniform Local Binary Patterns (LBP) (10 features)
- **Grayscale conversion**: Converted to 8-bit integer grayscale ($[0, 255]$ uint8) via `image.convert("L")`.
  Using integer grayscale is critical: floating-point representations introduce machine-epsilon jitter in neighbor
  comparisons that produce spurious pattern transitions.
- **Parameters**: Circular uniform LBP with $P = 8$ neighbors at radius $R = 1.0$ (`method="uniform"`).
- **Binning**: Uniform patterns have at most 2 bit transitions (0-to-1 or 1-to-0). For $P = 8$, there are exactly
  $P + 1 = 9$ uniform patterns (corresponding to 0 through 8 bit transitions) and 1 aggregate bin for all non-uniform
  patterns, totaling $P + 2 = 10$ bins.
- **Normalization**: The 10-bin histogram is L1-normalized to sum to 1.0, representing the empirical distribution of
  local surface texture across the image.

### 4. Fixed Concatenation Order
Descriptors are always concatenated in the fixed order:
$$\mathbf{x} = [\mathbf{x}_{\text{HSV}} \, (32) \,\|\, \mathbf{x}_{\text{HOG}} \, (1,764) \,\|\, \mathbf{x}_{\text{LBP}} \, (10)] \in \mathbb{R}^{1,806}$$
All feature values are unscaled raw floats (`float32`). Feature standardization (`StandardScaler`) must be fitted
strictly within cross-validation training folds by the modeling components to prevent test leakage.

---

## Reproducible Feature Caching & Stale Cache Detection

### CLI Commands
Teammates can generate or update the feature cache using the repository CLI:
```sh
# Generate feature cache (uses data/processed/features.npz by default)
waste-data features

# Alternatively, using the standalone entrypoint:
waste-features

# Force regeneration if modifying configurations:
waste-data features --force
```

### Cache Invariants & Data Integrity
1. **Audited Manifest Integration**: Reads `docs/dataset/manifest.csv`. Excluded images (the 6 conflicting-label
   images identified during data preparation) are **strictly filtered out**.
2. **Fixed Split Preservation**: Existing `split` column assignments (`train`, `validation`, `test`) and `duplicate_group`
   groupings are carried over directly from the manifest without regeneration.
3. **No Skipping Failed Images**: If any image cannot be opened, has a corrupted file hash, or fails extraction,
   the caching process aborts immediately with a clear error.
4. **Deterministic Ordering**: Rows are sorted deterministically by `image_id`, ensuring consistent alignment across runs.
5. **Stale Cache Detection**:
   The cache records:
   - `manifest_sha256`: SHA-256 of `docs/dataset/manifest.csv`
   - `config_hash`: Deterministic SHA-256 of the preprocessing and features configuration sections
   - `source_fingerprint`: SHA-256 hash of `features.py` and `preprocessing.py`
   If the manifest, configuration, or extractor source changes, `is_cache_stale()` marks the cache stale and
   `load_features(..., check_stale=True)` refuses to load it, preventing downstream models from training on desynchronized features.
6. **Safe Storage**: Saved with native NumPy string arrays so loading with `allow_pickle=False` works safely and securely.
   Stored under `data/processed/` which is ignored by Git.

---

## Verified Results

### 1. Synthetic Unit Tests (48 passing tests in ~1.5s)
Run with:
```sh
ruff check .
python -m unittest discover -s tests -v
```
Test coverage includes:
- Exact dimensions: HSV (32), HOG (1,764), LBP (10), Combined (1,806).
- Strict descriptor ordering and boundary slice matching.
- Dtype `float32`, all finite, no NaNs or Infinities.
- Histogram probability distribution normalization (HSV channel sums and LBP sum to 1.0).
- Deterministic extraction and independence from input mutation.
- Robust handling of RGB, grayscale, RGBA transparency, palette transparency, and EXIF orientations via shared preprocessing.
- Configuration validation and rejection of invalid parameters.
- Cache row alignment, exclusion filtering, and split preservation.
- Stale cache detection on manifest and configuration changes.
- CLI idempotency and `--force` regeneration.

### 2. Real Dataset Validation (TrashNet)
Executed `waste-data download` followed by `waste-data features`:
- Total images extracted: **2,521 usable images** (all 6 conflicting-label images excluded).
- Extracted matrix shape: `(2521, 1806)` of dtype `float32`.
- Memory footprint: ~18.2 MB uncompressed, ~10 MB compressed `.npz`.
- Execution time: **10.2 seconds** (~4.0 ms per image on local machine).
- Split distribution verified:
  - `train`: 1,764 images
  - `validation`: 380 images
  - `test`: 377 images
- Class counts verified:
  - `paper`: 594
  - `glass`: 498
  - `plastic`: 480
  - `metal`: 409
  - `cardboard`: 403
  - `trash`: 137
- Manifest SHA-256 verified against Om's data audit: `affd3e9d29e612b0179820a2a20ed21395e26d89a094f02bb7390d9400563314`.

---

## Limitations to Note for Modeling & Final Report

1. **Letterbox Padding Influence**: The aspect-ratio preserving letterbox resizes non-square images to fit within $128 \times 128$
   and pads borders with white (`[255, 255, 255]`). Global descriptors (HSV and LBP) include these white border pixels,
   which artificially skews Value towards 1.0 and adds uniform border patterns. This must be considered when evaluating
   model performance on unpadded real-world photos.
2. **Loss of Spatial Layout in Color**: Global HSV channel histograms discard spatial location (unlike spatial pyramid color histograms).
   A red object at the center produces the exact same color histogram as a red object at the perimeter.
3. **Downsampling to 128x128**: Downscaling $512 \times 384$ source images to $128 \times 128$ smooths fine surface textures (e.g., subtle
   cardboard corrugation or glass transparency reflections).
4. **Local Texture Window**: LBP at radius $R=1$ examines only adjacent 8-pixel neighborhoods, capturing fine micro-texture
   rather than macro-patterns.

---

## Viva Preparation & Discussion Topics

- **Why separate 1D channel histograms instead of a joint 3D HSV histogram?**
  A 3D histogram with $16 \times 8 \times 8 = 1,024$ bins for a $128 \times 128$ image ($16,384$ pixels) would lead to high
  sparsity and risk of overfitting in linear classifiers. 32 1D marginal bins compactly summarize color distributions.
- **Why use L2-Hys normalization for HOG?**
  Standard L2 normalization can be skewed by high-contrast edges or specular highlights. L2-Hys limits maximum gradient
  energies to 0.2 and renormalizes, providing robust illumination and contrast invariance.
- **Why must LBP be computed on uint8 grayscale instead of float?**
  Floating-point conversions can introduce tiny numerical precision noise ($10^{-16}$) between neighboring pixels of similar
  brightness. This flips the boolean threshold $(I_p \ge I_c)$, causing spurious pattern classifications. Integer grayscale is deterministic.
- **Why is feature caching decoupled from model scaling?**
  Features are purely deterministic functions of individual images. Data scaling (e.g. `StandardScaler`) computes dataset-level
  statistics ($\mu, \sigma$). Fitting scalers across the full dataset before splitting causes information leakage from validation/test
  sets into training.
- **Why is stale cache detection essential?**
  If dataset exclusions, split allocations, or image preprocessing are modified, training models on an outdated feature cache
  leads to invalid evaluation metrics and irreproducible experiments.
