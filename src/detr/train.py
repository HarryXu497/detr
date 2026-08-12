"""Training entry point for DETR.

Wires the model, Hungarian matcher, and set criterion together and runs the optimization
loop. The criterion returns a structured :class:`SetCriterionLoss`; :func:`weighted_loss`
reduces it to the single scalar that is actually optimized, applying the paper's loss
weights to the final decoder layer and every auxiliary layer alike. The backbone is
trained at a lower learning rate than the rest of the model, gradients are clipped, and
each epoch is checkpointed so a long run survives interruption.

Run as a module so the ``detr`` package resolves, e.g.::

    python -m detr.train --data /data/coco --ann .../instances.json --num-classes 80 \\
        --overfit 10 --epochs 200

The ``--overfit N`` path restricts training to the first ``N`` images and is the gate that
must drive the loss down before committing to a full run.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from detr.criterion import SetCriterion, SetCriterionLoss

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

from detr.dataset import COCODetectionDataset, VOCDetectionDataset, collate_fn
from detr.detr import DETR, DETROutputWithAuxOutputs
from detr.matcher import HungarianMatcher


def weighted_loss(loss: SetCriterionLoss, weight_dict: dict[str, float]) -> Tensor:
    """Reduce the structured criterion output to the scalar training objective.

    Applies ``weight_dict`` to the final-layer classification, L1, and GIoU terms, then
    adds the same weighted combination for each auxiliary decoder layer.

    Args:
        loss: the criterion output, holding the final-layer losses and per-layer
            auxiliary losses.
        weight_dict: maps ``"loss_ce"``, ``"loss_bbox"``, and ``"loss_giou"`` to their
            scalar weights. The same weights apply to every layer.

    Returns:
        The scalar loss to call ``backward`` on.
    """
    # Compute loss from final outputs
    total = (
        weight_dict["loss_ce"] * loss.loss_ce +
        weight_dict["loss_bbox"] * loss.loss_box.loss_bbox +
        weight_dict["loss_giou"] * loss.loss_box.loss_giou
    )

    # Add on loss from aux outputs
    for ce_aux, box_aux in zip(loss.loss_ce_aux, loss.loss_box_aux):
        total += (
            weight_dict["loss_ce"] * ce_aux +
            weight_dict["loss_bbox"] * box_aux.loss_bbox +
            weight_dict["loss_giou"] * box_aux.loss_giou
        )

    return total


def build_model_and_criterion(num_classes: int, device: str, num_queries: int = 100) -> tuple[DETR, SetCriterion]:
    """Construct the model and its criterion with the paper's loss weights.

    Args:
        num_classes: number of object classes, excluding the no-object class.
        device: device to place both modules on.
        num_queries: number of object queries (detection slots). Must exceed the largest
            object count in any single image.

    Returns:
        The DETR model and a :class:`SetCriterion` configured with the paper weights
        (``loss_ce=1, loss_bbox=5, loss_giou=2``) and its own Hungarian matcher.
    """
    detr = DETR(num_classes=num_classes, num_queries=num_queries).to(device)
    matcher = HungarianMatcher()
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict).to(device)

    return detr, criterion


def build_optimizer(model: DETR, lr: float = 1e-4) -> Optimizer:
    """Build the AdamW optimizer with a lower learning rate for the backbone.

    The pretrained backbone is fine-tuned gently at ``1e-5`` while the rest of the model
    trains at ``lr``; both share weight decay ``1e-4``.

    Args:
        model: the DETR model whose parameters are optimized.
        lr: learning rate for the non-backbone parameters. The backbone stays at ``1e-5``.

    Returns:
        An ``AdamW`` optimizer with two parameter groups split on the ``backbone`` prefix.
    """
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]

    return AdamW(
        [
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": 1e-5},
        ],
        weight_decay=1e-4
    )


def train_one_epoch(
    model: DETR,
    criterion: SetCriterion,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    weight_dict: dict[str, float],
    max_norm: float = 0.1
) -> dict[str, float]:
    """Run one pass over the data, updating the model, and return the loss breakdown.

    For each batch: move the images and targets to ``device``, compute the loss, reduce
    it with :func:`weighted_loss`, backpropagate, clip the gradient norm, and step the
    optimizer. The per-term losses are accumulated as plain floats for logging.

    Args:
        model: the DETR model in training mode.
        criterion: the set criterion.
        data_loader: yields ``(images, targets)`` from :func:`collate_fn`.
        optimizer: the optimizer to step.
        device: device to run on.
        weight_dict: loss weights passed through to :func:`weighted_loss`.
        max_norm: gradient-norm clip threshold.

    Returns:
        The epoch-averaged losses, with keys ``loss_ce``, ``loss_bbox``, ``loss_giou``,
        and ``total``. The component values are unweighted; ``total`` is the weighted
        objective.
    """
    total_loss = {
        "loss_ce": 0.0,
        "loss_bbox": 0.0,
        "loss_giou": 0.0,
        "total": 0.0
    }
    num_batches = 0

    for images, targets in data_loader:
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute outputs
        out: DETROutputWithAuxOutputs = model(images)
        loss: SetCriterionLoss = criterion(out, targets)
        total = weighted_loss(loss, weight_dict)

        # Step
        optimizer.zero_grad()
        total.backward()
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Accumulate losses for logging
        total_loss["loss_ce"] += loss.loss_ce.item()
        total_loss["loss_bbox"] += loss.loss_box.loss_bbox.item()
        total_loss["loss_giou"] += loss.loss_box.loss_giou.item()
        total_loss["total"] += total.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_loss.items()}


def save_checkpoint(path: str | Path, model: DETR, optimizer: Optimizer, scheduler: LRScheduler, epoch: int) -> None:
    """Write model, optimizer, scheduler, and epoch to ``path`` so a run can resume.

    Args:
        path: destination file. Its parent directory must already exist.
        model: model whose ``state_dict`` is saved.
        optimizer: optimizer whose ``state_dict`` is saved.
        scheduler: learning-rate scheduler whose ``state_dict`` is saved.
        epoch: the epoch just completed.
    """
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, path)


def load_checkpoint(
        path: str | Path, model: DETR, optimizer: Optimizer | None = None, scheduler: LRScheduler | None = None) -> int:
    """Restore a checkpoint into the given modules.

    The optimizer and scheduler are optional so the checkpoint can be loaded for
    inference or evaluation with the model alone.

    Args:
        path: checkpoint file written by :func:`save_checkpoint`.
        model: model to load weights into.
        optimizer: optimizer to restore, if given.
        scheduler: scheduler to restore, if given.

    Returns:
        The saved epoch. The training loop resumes from this value.
    """
    state_dict = torch.load(path)
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model"])

    if optimizer:
        optimizer.load_state_dict(state_dict["optimizer"])

    if scheduler:
        scheduler.load_state_dict(state_dict["scheduler"])

    return epoch


def main() -> None:
    """Parse arguments, build everything, and run the training loop.

    Supports ``--overfit N`` to restrict training to the first ``N`` images (the overfit
    gate), ``--resume`` to continue from a checkpoint, and ``--lr-drop`` to set the epoch
    at which the learning rate steps down. A checkpoint is written after every epoch.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)          # dataset root
    parser.add_argument("--dataset", choices=["coco", "voc"], default="voc")
    parser.add_argument("--ann", default=None)            # COCO annotation JSON (COCO only)
    parser.add_argument("--download", action="store_true")  # fetch VOC on first run (VOC only)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=200)
    parser.add_argument("--max-norm", type=float, default=0.1)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output-dir", default="runs/exp")
    parser.add_argument("--resume", default=None)         # path to a checkpoint
    parser.add_argument("--overfit", type=int, default=0)  # 0 = off; N = memorize N images
    parser.add_argument("--s3-sync", default=None)  # s3://bucket/prefix; None = off.
    args = parser.parse_args()

    if args.dataset == "coco" and args.ann is None:
        parser.error("--ann is required for --dataset coco")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, criterion = build_model_and_criterion(args.num_classes, device, num_queries=args.num_queries)
    optimizer = build_optimizer(model, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    dataset: Dataset
    if args.dataset == "coco":
        dataset = COCODetectionDataset(args.data, args.ann, image_size=512)
    else:
        dataset = VOCDetectionDataset(args.data, image_set="train", image_size=512,
                                      download=args.download)

    if args.overfit > 0:
        dataset = torch.utils.data.Subset(dataset, range(args.overfit))
        # TODO: need to turn off any image augmentation in the future (flips, crops, etc)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate_fn)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Create out directory if it does not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(model, criterion, data_loader, optimizer, device, criterion.weight_dict, max_norm=args.max_norm)
        scheduler.step()

        print(f"epoch {epoch}: {stats}")

        save_checkpoint(f"{args.output_dir}/checkpoint.pth", model, optimizer, scheduler, epoch)

        if args.s3_sync:
            subprocess.run(["aws", "s3", "sync", args.output_dir, args.s3_sync], check=False)


if __name__ == "__main__":
    main()
