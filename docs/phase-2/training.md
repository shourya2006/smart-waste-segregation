# Phase 2 training loop

**Owner:** Shourya Bafna
**Reviewer:** Daksh Batra
**Code:** [`src/waste_sorting/dl_training.py`](../../src/waste_sorting/dl_training.py)
**Configuration:** `[dl.training]` in [`configs/phase2.toml`](../../configs/phase2.toml)
**Tests:** [`tests/test_dl_training.py`](../../tests/test_dl_training.py)

The loop trains a pretrained backbone (frozen, or with its last blocks unfrozen) plus a
`Dropout -> Linear(1280, 6)` head. It reuses Om's loaders (`dl_data.make_dataloaders`) and
Aditya's backbones (`backbones.load_backbone`, `freeze`, `unfreeze_last_blocks`). The defaults
in `[dl.training]` are the **BASE** row of the pre-registered
[ablation plan](ablation-plan.md).

## 1. Split use

| Split | Used by this loop? | Purpose |
| --- | --- | --- |
| Training (1,764), 5 frozen folds | Yes | `cv`: train on 4 folds, early-stop on and score the 5th. `final`: train on all 1,764 |
| Validation (380) | **No** | Reserved for Prajjwal's calibration and abstention threshold |
| Test (377) | **No** | Reserved for the single final frozen evaluation |

`final` mode has no held-out data, so it cannot early-stop. It trains for a fixed number of
epochs: the `median_best_epoch` of the selected CV run.

## 2. Commands

Install as in [data-pipeline.md](data-pipeline.md). On Colab or Kaggle, check that
`torch.cuda.is_available()` prints `True` first. Each run needs a **new** output directory;
the loop refuses to overwrite an earlier run.

```sh
# Baseline 5-fold CV (BASE row of the ablation plan)
waste-dl-train cv --output artifacts/phase2/runs/base

# One-factor ablations: change exactly one setting with --set (repeatable)
waste-dl-train cv --output artifacts/phase2/runs/a1.1 --set dl.training.backbone=efficientnet_b0
waste-dl-train cv --output artifacts/phase2/runs/a2.1 --set dl.training.unfreeze_blocks=2
waste-dl-train cv --output artifacts/phase2/runs/a3.1 --set preprocessing.width=160 --set preprocessing.height=160
waste-dl-train cv --output artifacts/phase2/runs/a4.2 --set dl.training.dropout=0.5
waste-dl-train cv --output artifacts/phase2/runs/a5.1 --set dl.augmentation.enabled=false
waste-dl-train cv --output artifacts/phase2/runs/a6.1 --set dl.training.class_weighted_loss=true
waste-dl-train cv --output artifacts/phase2/runs/a6.2 --set dl.data.balanced_sampling=true

# After selection: train the chosen configuration on the whole training split
waste-dl-train final --epochs <median_best_epoch> --output artifacts/phase2/final/<name> [--set ...]
```

`--set` parses values as TOML (`0.5`, `true`, `160`); anything else is kept as text, so
`backbone=efficientnet_b0` works without quotes. An unknown key is an error, so a typo can't
silently run the baseline instead. Setting both `class_weighted_loss` and `balanced_sampling`
is rejected.

## 3. Outputs

| File | Mode | Contents |
| --- | --- | --- |
| `runs.csv` | `cv` | One row per fold: `macro_f1`, `accuracy`, `recall_<class>` x 6, `val_loss`, `best_epoch`, `epochs_run`, `train_seconds` |
| `oof_predictions.csv` | `cv` | One row per training image, scored by the fold model that never trained on it: `image_id`, `fold`, `label`, `prob_<class>` x 6. Daksh's late-fusion hybrid uses these |
| `summary.json` | `cv` | Mean ± std macro F1 (sample std), mean accuracy, `median_best_epoch`, trainable parameters, device, the full resolved config, and source provenance (git revision, dirty flag, package hash, torch versions) |
| `model.pt`, `metadata.json` | `final` | State dict, plus config, epochs, history, SHA-256 of `model.pt` and provenance |

`runs.csv` and `oof_predictions.csv` are rewritten after every fold, so a crash keeps the
finished folds. Only a run with `summary.json` is complete. Copy the CSV and JSON files into `docs/phase-2/ablations/` when reporting. Checkpoints stay
in `artifacts/`, which Git ignores.

For Prajjwal and Saksham, load a final model with:

```python
from waste_sorting.dl_training import load_trained_model

model, metadata = load_trained_model("artifacts/phase2/final/<name>")  # eval mode, hash checked
```

## 4. What happens in one epoch

1. `set_train_mode(model)`: the head and any unfrozen blocks go to train mode, and every
   fully frozen block of `model.features` stays in **eval** mode.
2. For each training batch: forward pass, cross-entropy (class-weighted only if
   `class_weighted_loss`), backward pass, AdamW step.
3. Cosine schedule step: the learning rate falls from its start value to `lr_min` over
   `max_epochs`. This holds in `final` mode too, so a final run of N epochs follows exactly
   the schedule of a CV fold that early-stopped at epoch N. `--epochs` may not exceed
   `max_epochs`.
4. `cv` only: compute the **unweighted** cross-entropy on the held-out fold. When it improves,
   keep a copy of the weights. After `patience` epochs without improvement, stop and restore
   the best copy.

The held-out loss is always unweighted, so early stopping follows the same rule in every
ablation, including the class-weighted one.

### Why frozen blocks stay in eval mode

`requires_grad = False` stops gradients, but not train-mode behaviour:

- **BatchNorm** keeps updating its running mean and variance in train mode. Aditya's
  `keep_frozen_bn_eval` handles this.
- **EfficientNet-B0's stochastic depth** randomly drops residual branches in train mode, even
  in blocks whose weights are frozen. `keep_frozen_bn_eval` does not cover this.

`set_train_mode` puts each fully frozen block in eval mode, which covers both. A frozen
backbone is then a fixed feature extractor, the same as the one behind the cached embeddings.
Tests check both: a frozen EfficientNet gives identical outputs on repeated train-mode forward
passes, and none of its weights or BatchNorm statistics change during training.

### Parameter groups

| Setting | Trainable parameters | Optimiser groups |
| --- | --- | --- |
| Frozen (`unfreeze_blocks = 0`) | 7,686 (head only) | head at `lr_head = 1e-3` |
| `unfreeze_blocks = n` | head + `model.features[-n:]` (counts in the [ablation plan](ablation-plan.md)) | head at `1e-3`, unfrozen blocks at `lr_backbone = 1e-4` |

## 5. Smoke check (not a result)

This run only checked that the loop works end to end on the real data. It is **not** an
ablation result and must not be reported as one.

Settings: `waste-dl-train cv --set dl.training.max_epochs=3` on an Apple M-series GPU (MPS),
frozen MobileNetV2.

- Mean CV macro F1 after 3 epochs: 0.754 ± 0.027 (still improving at epoch 3).
- About 14 s per epoch.
- All 1,764 training images received exactly one out-of-fold prediction.

The pre-registered BASE run (up to 40 epochs, patience 8) is Aditya's first ablation run.

## 6. Reproducibility notes

- `torch.manual_seed(seed)` runs before each fold and before the final run. Loaders use
  seeded generators and seeded workers (`dl_data`).
- CPU runs repeat exactly. GPU and MPS kernels are not bit-for-bit deterministic, so expect
  small differences between machines. Record the `device` in `summary.json` next to every
  reported number.
- Provenance comes from the code checkout. `working_tree_dirty: true` means uncommitted
  code; commit before any run you plan to report.

## 7. Theory for the viva

- **Cross-entropy:** `-log p(correct class)`. It is minimised by calibrated probabilities and
  punishes confident mistakes heavily. Class weights multiply each example's loss by
  `n / (k * n_c)`, so mistakes on rare classes (trash: 96 training images) cost more.
- **AdamW:** Adam with *decoupled* weight decay. The decay shrinks the weights directly
  instead of being added to the gradient, where Adam's per-parameter scaling would weaken
  it. This makes it a real L2-style regulariser (Loshchilov & Hutter, 2019).
- **Cosine annealing:** the learning rate follows half a cosine from the start value to
  `lr_min`. Steps are large early, then small near the end, so training settles into a
  minimum instead of bouncing around it (Loshchilov & Hutter, 2017).
- **Early stopping:** held-out loss falls and then rises as the model starts to memorise.
  Stopping at the minimum and restoring those weights acts as a regulariser
  (Prechelt, 1998).
- **Differential learning rates:** pretrained layers already hold useful ImageNet features.
  A 10x smaller learning rate lets them adapt without wiping out what they learned
  (*catastrophic forgetting*). The new head starts from random weights, so it needs the
  larger rate (Howard & Ruder, 2018).
- **Fine-tuning only late blocks:** early layers detect generic edges and textures that
  transfer well. Late layers are the most ImageNet-specific, so those are the ones worth
  adapting (Yosinski et al., 2014).
- **BatchNorm in eval mode:** normalises with stored population statistics, not batch
  statistics (Ioffe & Szegedy, 2015). Batches of 32 TrashNet photos would replace ImageNet's
  statistics with noisy estimates.
- **Stochastic depth:** randomly skips residual branches during training as a regulariser
  (Huang et al., 2016). In a frozen block it adds noise without any learning, so it is turned
  off there.

### Likely viva questions

1. *Why not early-stop on the validation split?* It is reserved for calibration and the
   abstention threshold. Using it for stopping too would make those estimates optimistic.
2. *How many epochs does the final model train, and why?* The median best epoch across the
   5 CV folds. The final run has no held-out data to stop on.
3. *Why is the held-out loss unweighted when the training loss is weighted?* So the stopping
   rule is identical across ablations. The weighting is a training choice, not a different
   definition of a good model.
4. *What does `set_train_mode` add over `keep_frozen_bn_eval`?* It also turns off stochastic
   depth in frozen EfficientNet blocks.
5. *Why AdamW and not Adam with `weight_decay`?* In Adam, weight decay is folded into the
   gradient and rescaled per parameter, so it no longer acts as true L2 decay. AdamW applies
   it directly.
6. *Why keep out-of-fold probabilities?* Each one comes from a model that never saw that
   image, so a late-fusion weight can be chosen on them without leakage.
7. *Is a 0.02 difference in macro F1 meaningful?* Only if it is larger than the fold standard
   deviation. That is the rule in the ablation plan.
8. *Why do GPU runs differ slightly from CPU runs?* Parallel floating-point reductions on GPU
   and MPS are not bit-for-bit deterministic.

## 8. Report paragraph (Training section)

```latex
\subsection{Training Procedure}
Each backbone is followed by a dropout layer ($p=0.2$) and a linear classifier over the six
classes. Models are optimised with AdamW~\cite{loshchilov2019adamw} (weight decay $10^{-2}$)
under a cosine learning-rate schedule~\cite{loshchilov2017sgdr} from $10^{-3}$ to $10^{-5}$.
When late backbone blocks are fine-tuned, they use a ten-times smaller learning rate than the
new head, following discriminative fine-tuning~\cite{howard2018ulmfit}, so that pretrained
features are adapted rather than overwritten~\cite{yosinski2014transferable}. Frozen blocks are
kept in inference mode, which fixes their batch-normalisation statistics~\cite{ioffe2015batchnorm}
and disables stochastic depth~\cite{huang2016stochasticdepth}, so that a frozen backbone is a
fixed feature extractor. Model selection uses the same five group-aware training folds as
Phase~1: each fold model is early-stopped~\cite{prechelt1998early} on the unweighted
cross-entropy of its held-out fold (patience 8, at most 40 epochs) and scored by macro F1. The
final model is retrained on the full training split for the median best epoch across folds.
The validation split is reserved for calibration and the abstention threshold, and the test
split is used once, for the frozen evaluation.
```
