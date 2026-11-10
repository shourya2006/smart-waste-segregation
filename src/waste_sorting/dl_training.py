"""Phase 2 training loop: pretrained backbone (frozen or partly fine-tuned) + linear head.

Settings come from ``[dl.training]`` in configs/phase2.toml and follow the pre-registered
docs/phase-2/ablation-plan.md.

- ``cv``: for each frozen Phase 1 fold, train on the other folds, early-stop on the held-out
  fold, and record its scores and out-of-fold probabilities.
- ``final``: train on the whole training split for a fixed number of epochs (take it from
  the selected CV run's ``median_best_epoch``).

Neither mode scores the validation or test split. Validation is reserved for calibration
and the abstention threshold; test is reserved for the single final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import statistics
import time
import tomllib
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn

from waste_sorting import dl_data
from waste_sorting.backbones import (
    count_trainable_parameters,
    freeze,
    load_backbone,
    unfreeze_last_blocks,
)
from waste_sorting.training import _source_metadata


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_overrides(config, items):
    """Return a copy of ``config`` with dotted ``key=value`` overrides applied.

    Values are parsed as TOML (``0.5``, ``true``, ``[160, 160]``); anything else is kept as
    a string. Unknown keys are rejected so a typo cannot silently run the baseline.
    """
    config = copy.deepcopy(config)
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value: {item!r}")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        try:
            node[leaf] = tomllib.loads(f"value = {raw}")["value"]
        except tomllib.TOMLDecodeError:
            node[leaf] = raw
    return config


def build_model(config, pretrained=True):
    """``nn.Sequential(backbone, head)``; only the head and unfrozen blocks train."""
    settings = config["dl"]["training"]
    backbone, dim = load_backbone(settings["backbone"], pretrained=pretrained)
    freeze(backbone)
    unfreeze_last_blocks(backbone, settings["unfreeze_blocks"])
    head = nn.Sequential(nn.Dropout(settings["dropout"]), nn.Linear(dim, len(config["classes"])))
    return nn.Sequential(backbone, head)


def set_train_mode(model):
    """Train mode for the head and unfrozen blocks; frozen blocks stay in eval mode.

    Eval mode on a frozen block keeps its BatchNorm running statistics fixed (what
    ``keep_frozen_bn_eval`` does) and also switches off EfficientNet's stochastic depth
    there, so frozen layers behave as a fixed feature extractor, as in the cached embeddings.
    """
    model.train()
    for block in model[0].features:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()


def make_optimizer(model, settings):
    """AdamW: head at ``lr_head``; unfrozen backbone blocks (if any) at ``lr_backbone``."""
    groups = [{"params": list(model[1].parameters()), "lr": settings["lr_head"]}]
    backbone = [p for p in model[0].parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": settings["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])


@torch.no_grad()
def predict(model, loader, device):
    """Return ``(labels, probabilities, image_ids, mean unweighted cross-entropy)``."""
    model.eval()
    labels, logits, image_ids = [], [], []
    for images, targets, ids in loader:
        logits.append(model(images.to(device)).float().cpu())
        labels.append(targets)
        image_ids.extend(ids)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = nn.functional.cross_entropy(logits, labels).item()
    return labels.numpy(), logits.softmax(dim=1).numpy(), image_ids, loss


def score(labels, probabilities, classes):
    """Macro F1, accuracy and per-class recall, the Phase 1 reporting metrics."""
    predicted = probabilities.argmax(axis=1)
    index = list(range(len(classes)))
    recall = recall_score(labels, predicted, labels=index, average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(labels, predicted, labels=index, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, predicted)),
        **{f"recall_{name}": float(r) for name, r in zip(classes, recall, strict=True)},
    }


def train_model(model, config, train_loader, validation_loader=None, *, epochs=None, device,
                log=print):
    """Train in place. With a validation loader, stop early on its loss and restore the best
    epoch's weights; without one, run exactly ``epochs`` epochs."""
    settings = config["dl"]["training"]
    if settings["class_weighted_loss"] and config["dl"]["data"]["balanced_sampling"]:
        raise ValueError("Use class_weighted_loss or balanced_sampling, not both")
    if validation_loader is None and not epochs:
        raise ValueError("Without a validation loader, pass a fixed number of epochs")
    weight = None
    if settings["class_weighted_loss"]:
        labels = train_loader.dataset.labels
        weight = dl_data.class_weights(labels, len(config["classes"])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    if epochs and epochs > settings["max_epochs"]:
        raise ValueError(f"epochs must be <= max_epochs ({settings['max_epochs']})")
    max_epochs = epochs or settings["max_epochs"]
    optimizer = make_optimizer(model, settings)
    # Always anneal over max_epochs: a final run of N epochs then follows the same schedule
    # as a CV fold that early-stopped at epoch N.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["max_epochs"], eta_min=settings["lr_min"]
    )
    # best_epoch starts at 0 with validation so a NaN loss still triggers early stopping.
    best_epoch = max_epochs if validation_loader is None else 0
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, max_epochs + 1):
        set_train_mode(model)
        total, count = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": total / count}
        scheduler.step()
        if validation_loader is not None:
            row["val_loss"] = predict(model, validation_loader, device)[3]
            if row["val_loss"] < best_loss:
                best_loss, best_epoch = row["val_loss"], epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        log("  epoch " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in row.items()))
        if validation_loader is not None and epoch - best_epoch >= settings["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history)}


def _run_metadata(config, root, device, model):
    return {
        "config": config,
        "device": device.type,
        "trainable_parameters": count_trainable_parameters(model),
        "source": {
            **_source_metadata(root),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
        },
    }


def run_cv(config, root, output, *, pretrained=True, log=print):
    """Train and score every frozen fold; write runs.csv, oof_predictions.csv, summary.json."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    classes = config["classes"]
    seed = config["dl"]["data"]["seed"]
    n_splits = json.loads((root / config["paths"]["folds"]).read_text())["n_splits"]
    device = pick_device()
    rows, predictions = [], []
    for fold in range(n_splits):
        log(f"fold {fold + 1}/{n_splits} on {device.type}")
        torch.manual_seed(seed)
        loaders = dl_data.make_dataloaders(config, root, fold=fold)
        model = build_model(config, pretrained).to(device)
        start = time.perf_counter()
        result = train_model(model, config, loaders["train"], loaders["validation"],
                             device=device, log=log)
        seconds = time.perf_counter() - start
        labels, probabilities, image_ids, loss = predict(model, loaders["validation"], device)
        rows.append({
            "fold": fold,
            **score(labels, probabilities, classes),
            "val_loss": loss,
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "train_seconds": round(seconds, 1),
        })
        log(f"fold {fold} macro_f1={rows[-1]['macro_f1']:.4f} best_epoch={result['best_epoch']}")
        frame = pd.DataFrame({"image_id": image_ids, "fold": fold, "label": labels})
        for i, name in enumerate(classes):
            frame[f"prob_{name}"] = probabilities[:, i]
        predictions.append(frame)
        # Written after every fold so a crash or Colab disconnect keeps the finished folds.
        pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
        pd.concat(predictions).sort_values("image_id").to_csv(
            output / "oof_predictions.csv", index=False
        )
    runs = pd.DataFrame(rows)
    summary = {
        "mode": "cv",
        "selection_metric": "mean fold macro F1",
        "mean_macro_f1": float(runs["macro_f1"].mean()),
        "std_macro_f1": float(runs["macro_f1"].std(ddof=1)) if len(runs) > 1 else 0.0,
        "mean_accuracy": float(runs["accuracy"].mean()),
        "median_best_epoch": int(statistics.median_low(runs["best_epoch"])),
        **_run_metadata(config, root, device, model),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def train_final(config, root, output, epochs, *, pretrained=True, log=print):
    """Train on the whole training split for ``epochs`` epochs; save model.pt and metadata.

    Nothing is scored here: calibration, threshold and the test evaluation happen later.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["dl"]["data"]["seed"])
    device = pick_device()
    train_loader = dl_data.make_dataloaders(config, root)["train"]
    model = build_model(config, pretrained).to(device)
    result = train_model(model, config, train_loader, epochs=epochs, device=device, log=log)
    path = output / "model.pt"
    torch.save(model.cpu().state_dict(), path)
    metadata = {
        "mode": "final",
        "epochs": epochs,
        "classes": config["classes"],
        "history": result["history"],
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **_run_metadata(config, root, device, model),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def load_trained_model(directory):
    """Rebuild a ``final`` model from its directory and verify the checkpoint hash."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    path = directory / "model.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError(f"Checkpoint hash does not match metadata: {path}")
    model = build_model(metadata["config"], pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval(), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Phase 2 transfer-learning model.")
    parser.add_argument("mode", choices=["cv", "final"])
    parser.add_argument("--output", required=True, help="New directory for this run")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--config", default="configs/phase2.toml", help="Path under --root")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="e.g. dl.training.dropout=0.5 (repeatable)")
    parser.add_argument("--epochs", type=int, help="final mode: fixed epoch count")
    args = parser.parse_args(argv)
    root = Path(args.root)
    config = apply_overrides(dl_data.load_config(root / args.config), args.overrides)
    if args.mode == "cv":
        summary = run_cv(config, root, args.output)
        print(f"mean macro F1 {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f}; "
              f"median best epoch {summary['median_best_epoch']}")
    else:
        if not args.epochs:
            parser.error("final mode needs --epochs (use the CV run's median_best_epoch)")
        train_final(config, root, args.output, args.epochs)
        print(f"Saved {Path(args.output) / 'model.pt'}")


if __name__ == "__main__":
    main()
