# Phase 2 data pipeline and regularisation

Owner: Om Yadav. Code: [`src/waste_sorting/dl_data.py`](../../src/waste_sorting/dl_data.py),
settings: [`configs/phase2.toml`](../../configs/phase2.toml), tests:
[`tests/test_dl_data.py`](../../tests/test_dl_data.py).

This component turns the frozen Phase 1 dataset (manifest, splits and CV folds) into
PyTorch `DataLoader`s for the MobileNetV2 / EfficientNet-B0 backbones, the training
loop and the hybrids. It adds no new split, statistic or image cache.

## Setup

```sh
python -m pip install -r requirements-demo.lock
python -m pip install -r requirements-dl.lock
python -m pip install --no-deps -e .
waste-dl-data          # or: python -m waste_sorting.dl_data
```

Only CI and Linux machines without an NVIDIA GPU should add
`--extra-index-url https://download.pytorch.org/whl/cpu` to the `requirements-dl.lock`
line; it installs the smaller `+cpu` wheels, which cannot use a GPU.

### GPU machines (Colab / Kaggle)

Install `requirements-dl.lock` **without** `--extra-index-url`; the default PyPI Linux
wheels include CUDA. Then check:

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It must print `True`. If the version ends in `+cpu`, reinstall `torch` and `torchvision`
without the CPU index. macOS has no CUDA: training there uses the CPU (or MPS, if a
teammate adds it later).

`waste-dl-data` prints split sizes and class counts and saves
`artifacts/phase2/augmentation_preview.png` (training images only; ignored by Git).
Expected sizes, which match `docs/dataset/split_summary.json`:

| Split | Images | cardboard | glass | metal | paper | plastic | trash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1764 | 282 | 348 | 286 | 416 | 336 | 96 |
| validation | 380 | 61 | 75 | 62 | 89 | 72 | 21 |
| test | 377 | 60 | 75 | 61 | 89 | 72 | 20 |

## Interface for teammates

| Function | Arguments | Returns |
| --- | --- | --- |
| `load_config(path)` | TOML path | config `dict` |
| `make_dataloaders(config, root=".", *, fold=None)` | config, repository root, optional CV fold `0..4` | `{"train", "validation", "test"}` loaders; with `fold=k`, only `{"train", "validation"}` |
| `WasteImageDataset(manifest, image_root, classes, transform, *, split=None, image_ids=None)` | manifest path or DataFrame; exactly one of `split` or `image_ids` (all from one split) | items `(tensor[3,224,224] float32, label_index, image_id)` in image-ID order; `.labels`, `.image_ids`, `.classes`, `.split` |
| `build_transforms(config, train)` | config, `True` for training | letterbox → augmentation (train only) → float tensor → ImageNet normalisation |
| `fold_image_ids(fold, folds_path)` | fold `0..4`, `docs/training/folds.json` | `(train_ids, validation_ids)`, sorted lists |
| `class_weights(labels, num_classes)` | training labels | `float32` tensor `n / (k · n_c)` |
| `make_balanced_sampler(labels, num_classes, generator=None)` | training labels | `WeightedRandomSampler` with equal class probability |
| `save_augmentation_preview(dataset, config, path, variants=5)` | a **training** dataset | PNG path |

```python
import torch
from waste_sorting.dl_data import class_weights, load_config, make_dataloaders

config = load_config("configs/phase2.toml")
torch.manual_seed(config["dl"]["data"]["seed"])
loaders = make_dataloaders(config, ".")          # or fold=k for 5-fold CV
weights = class_weights(loaders["train"].dataset.labels, len(config["classes"]))
for images, labels, image_ids in loaders["train"]: ...
```

Notes:

- Class index `i` is `config["classes"][i]`, identical to `configs/phase1.toml`.
- Call `torch.manual_seed(seed)` once at the start of a run. With workers, each
  worker's augmentation stream is seeded from the loader's seeded generator; with
  `num_workers = 0`, augmentation uses the main-process torch seed.
- On macOS and Windows, workers are spawned, so training scripts need an
  `if __name__ == "__main__":` guard.
- For ablations, copy the config and change `[dl.augmentation]` or
  `[dl.data].balanced_sampling`; `enabled = false` makes the train transform equal to the
  evaluation transform.

## Augmentation: what and why it is label-preserving

Augmentation applies to training images only. Validation, test and (later) the robustness
set always use the deterministic evaluation transform. Every image is first letterboxed to
224×224 by the shared Phase 1 `preprocess_image` (EXIF orientation, RGB, aspect-preserving
LANCZOS resize, white padding), so Phase 1 and Phase 2 see the same geometry.

| Step | Setting | Why the label does not change |
| --- | --- | --- |
| Random resized crop | area 0.8–1.0, aspect 3/4–4/3 | At least 80% of the frame is kept, so the object stays in view; it simulates small changes in camera distance and framing. |
| Horizontal flip | p = 0.5 | A mirrored bottle, can or box is the same material. |
| Rotation | ±10°, white fill | Hand-held photos are rarely level. Exposed corners take the letterbox colour, so no black wedges appear that the model could learn as an artefact. |
| Colour jitter | brightness ±15%, contrast ±15%, saturation ±10%, hue 0 | Simulates mild lighting changes without changing which colour the object is. |

What is deliberately left out:

- **Vertical flips and large rotations.** Objects in TrashNet are photographed upright on a
  board with light from above. Upside-down images introduce shading and label orientation
  never seen at test time, and add no realistic variation.
- **Hue shifts, grayscale, strong saturation or contrast changes.** Material is recognised
  largely by colour and gloss: brown cardboard vs white paper, green/brown glass, specular
  metal highlights, translucent plastic. Changing hue can turn brown glass into green glass
  or cardboard into something paper-like, which would teach the network wrong cues.
- **Cutout, mixup, heavy blur or noise.** These are useful regularisers, but they are not
  conservative and they would make the augmentation ablation harder to interpret. They can
  be added later as separate, switchable steps.

The preview grid (`waste-dl-data`) shows one training image per class with five augmented
versions; every version should still clearly show the same material.

## Augmentation as a regulariser

TrashNet's training split is small (1764 images) compared with the parameters of even a
compact CNN, so a fine-tuned backbone can memorise individual photos. Augmentation enlarges
the effective training distribution with plausible variants of each image. This makes it
harder to fit noise and encodes the invariances we want (small shifts, scale, mirror,
lighting) without changing the model or the loss. In that sense it is a regulariser, like
weight decay or dropout, but it acts on the data rather than the weights
(Shorten and Khoshgoftaar, 2019). Every step has an on/off switch so Aditya's ablation
can measure the effect of augmentation on cross-validated macro F1 instead of assuming it.

## Class imbalance: sampler vs class weights

The training split is imbalanced: paper has 416 images and trash 96 (4.3×). An
unweighted loss lets the network spend most of its effort on paper, glass and plastic and
under-predict trash, which hurts macro F1, the project's selection metric.

| Option | How | Trade-off |
| --- | --- | --- |
| No correction | `balanced_sampling = false`, plain loss | Every image is seen exactly once per epoch; biased towards large classes. |
| Balanced sampler | `balanced_sampling = true` | Classes appear equally often in each batch. Minority images repeat often (about 4× for trash), but each repeat gets a different augmentation. Some majority images may be skipped in a given epoch. |
| Class-weighted loss | `CrossEntropyLoss(weight=class_weights(...))` | Every image is seen once per epoch; trash errors cost about 3.1× and paper errors about 0.71×. Effective batch loss scale varies with batch composition. |

The weights from the current training split are cardboard 1.04, glass 0.84, metal 1.03,
paper 0.71, plastic 0.88 and trash 3.06. Buda et al. (2018) found that oversampling
generally works well against class imbalance in CNNs and, unlike in classical ML, does not
cause severe overfitting. Use **one** correction, not both, because combining them
over-corrects towards the minority class. The config default is `false` (no correction)
so the baseline is simple; the choice should be made on the CV folds, as Phase 1 did for
`class_weight`. Weights must always be computed from training labels (or the fold's
training labels in CV), never from validation or test.

## Data-leakage safeguards inherited from Phase 1

- **Frozen splits.** Only rows with `included == True` are used and the committed `split`
  column is read as is. Nothing in this module can create or change a split.
- **Grouped duplicates.** `load_manifest` rejects a manifest where any `duplicate_group`
  crosses splits, so near-identical photos cannot sit in train and test.
- **Frozen folds.** `fold_image_ids` reads `docs/training/folds.json`; `make_dataloaders`
  checks that the folds exactly partition the training split, and returns no test loader in
  fold mode.
- **No statistics from held-out data.** Normalisation uses the fixed ImageNet mean and
  standard deviation that pretrained backbones expect (Deng et al., 2009); no mean, standard
  deviation, augmentation choice or class weight is computed from validation or test pixels.
- **Train-only augmentation.** Validation and test loaders use the deterministic
  evaluation transform; `save_augmentation_preview` refuses non-training datasets.
- **Single-split selections.** `WasteImageDataset(image_ids=...)` rejects excluded rows and
  IDs from more than one split.
- **No cached tensors.** Images are decoded on the fly; nothing derived from the data is
  written except the ignored preview PNG.

## Viva notes

1. **Why reuse `preprocess_image` instead of `torchvision.Resize`?** One letterbox for
   Phase 1, Phase 2 and the demo means identical EXIF handling, aspect ratio and white
   padding; a second resizer could silently change geometry or stretch objects.
2. **Why ImageNet mean/std instead of TrashNet statistics?** The backbones were pretrained
   with those statistics, and using fixed constants means nothing is estimated from data,
   so validation/test cannot leak into preprocessing.
3. **Why is augmentation applied only to training images?** Evaluation must measure
   performance on the real distribution with a deterministic input; random transforms at
   evaluation would make scores noisy and no longer comparable with Phase 1.
4. **Why no vertical flip or hue jitter?** Upside-down objects never appear in the data,
   and hue/strong colour changes destroy the main material cue (colour and gloss) and can
   make one class look like another.
5. **How is augmentation a regulariser?** It increases the variety of training inputs so
   the network cannot memorise individual photos, encoding invariances (shift, scale,
   mirror, lighting) that reduce the gap between training and validation performance.
6. **How do you handle class imbalance?** Either a `WeightedRandomSampler` with weight
   `1/n_c` per image (equal class probability) or inverse-frequency loss weights
   `n / (k · n_c)`, computed from training labels only. Not both at once.
7. **Sampler vs weighted loss, which is better?** It is an empirical question for our data;
   oversampling is often strong for CNNs (Buda et al., 2018), and with augmentation the
   repeated trash images are not identical. We choose on CV macro F1.
8. **How do you make runs reproducible?** Fixed seed 42, a seeded `torch.Generator` for
   shuffling and sampling, worker seeds derived from it (`seed_worker` also seeds NumPy and
   `random`), and `torch.manual_seed` at the start of a run.
9. **How does CV work in Phase 2?** `fold_image_ids(k)` reproduces the frozen Phase 1
   `StratifiedGroupKFold` assignments: train on four folds, validate on fold k. The test set
   is not touched during CV.
10. **What stops leakage between splits?** Committed splits with grouped near-duplicates, a
    check that no duplicate group crosses splits, excluded rows dropped, single-split ID
    selections, and tests asserting that no image ID appears in two splits.

## Report paragraph (Dataset & Regularisation)

```latex
\subsection{Deep-Learning Data Pipeline and Regularisation}
Phase~2 reuses the frozen Phase~1 splits (1764/380/377 training/validation/test images)
and the five group-aware cross-validation folds without modification. Each image is
EXIF-corrected, converted to RGB and letterboxed to $224\times224$ pixels with white
padding by the same function used in Phase~1, then normalised with the ImageNet channel
statistics expected by the pretrained backbones~\cite{deng2009imagenet}; no statistic is
estimated from validation or test images. As a data-level regulariser, training images
only receive conservative, label-preserving augmentation~\cite{shorten2019survey}: random
resized crops covering 80--100\% of the image, horizontal flips, rotations of up to
$\pm10^\circ$ with white fill, and mild brightness, contrast and saturation jitter. Vertical
flips and hue changes are excluded because material classes are recognised largely by
colour and gloss. The training split is imbalanced (96 \emph{trash} versus 416
\emph{paper} images); following evidence that oversampling is effective for convolutional
networks~\cite{buda2018imbalance}, the pipeline supports either class-balanced sampling or
inverse-frequency class weights computed from training labels, with the choice made on
cross-validated macro-F1.
```

## References

- C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep
  learning," *Journal of Big Data*, vol. 6, no. 60, 2019. doi:10.1186/s40537-019-0197-0
- M. Buda, A. Maki and M. A. Mazurowski, "A systematic study of the class imbalance
  problem in convolutional neural networks," *Neural Networks*, vol. 106, pp. 249–259,
  2018. doi:10.1016/j.neunet.2018.07.011
- J. Deng et al., "ImageNet: A large-scale hierarchical image database," *CVPR*, 2009,
  pp. 248–255. doi:10.1109/CVPR.2009.5206848

BibTeX keys in `report/references.bib`: `shorten2019survey`, `buda2018imbalance`,
`deng2009imagenet`.
