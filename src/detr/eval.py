"""Mean average precision evaluation for DETR.

Loads a trained checkpoint and reports COCO-style mAP over the VOC validation split, using
:func:`detr.postprocess.postprocess` to decode predictions and
:class:`torchmetrics.detection.MeanAveragePrecision` (pycocotools backend) to score them.
Run as a module::

    python -m detr.eval --checkpoint runs/voc/checkpoint.pth --data /data/voc \\
        --num-classes 20 --num-queries 100

``--num-classes`` and ``--num-queries`` must match the checkpoint.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
import argparse

from detr.box_ops import box_cxcywh_to_xyxy
from detr.dataset import VOCDetectionDataset, collate_fn
from detr.detr import DETR, DETROutputWithAuxOutputs
from detr.postprocess import postprocess
from detr.train import load_checkpoint


@torch.no_grad()
def evaluate(model: DETR, data_loader: DataLoader, device: str) -> dict:
    """Compute COCO-style mean average precision over a dataset.

    Runs the model on each batch, decodes predictions with
    :func:`detr.postprocess.postprocess`, and accumulates them against the ground truth in
    :class:`torchmetrics.detection.MeanAveragePrecision`. Everything is evaluated in
    normalized ``xyxy`` space (predictions via ``target_sizes`` of ones, ground truth via
    ``cxcywh``-to-``xyxy``); IoU is scale-invariant, so this needs no original image sizes.
    All predictions are passed with their scores and no threshold, because AP integrates
    over score thresholds itself.

    Args:
        model: the trained DETR model.
        data_loader: yields ``(images, mask, targets)`` from :func:`detr.dataset.collate_fn`.
        device: device to run the forward pass on. Metric accumulation is on CPU.

    Returns:
        The ``MeanAveragePrecision`` result dict, including ``map`` (mAP over IoU
        ``0.5:0.95``) and ``map_50`` (mAP at IoU 0.5, the classic VOC number).
    """
    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()

    for images, mask, targets in data_loader:
        images = images.to(device)
        mask = mask.to(device)
        out: DETROutputWithAuxOutputs = model(images, mask)
        # Normalized space
        # (B, 2) of ones instead of W, H
        sizes = torch.ones(images.shape[0], 2, device=device)
        detections = postprocess(out.output, sizes)

        preds = [
            {
                "scores": detection.scores.cpu(),
                "labels": detection.labels.cpu(),
                "boxes": detection.boxes.cpu()
            } for detection in detections
        ]

        gts = [
            {
                "labels": target["labels"].cpu(),
                "boxes": box_cxcywh_to_xyxy(target["boxes"]).cpu()
            } for target in targets
        ]

        metric.update(preds, gts)

    return metric.compute()


def main() -> None:
    """Load a checkpoint and print VOC-val mAP.

    Builds the model to match the checkpoint (``num_classes``/``num_queries`` must agree),
    evaluates over the VOC validation split, and prints ``map_50`` (mAP at IoU 0.5) and
    ``map`` (mAP over IoU 0.5:0.95).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--data")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DETR(num_classes=args.num_classes, num_queries=args.num_queries, dropout=args.dropout)
    load_checkpoint(args.checkpoint, model)

    model.to(device)
    model.eval()

    dataset = VOCDetectionDataset(args.data, image_set="val", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate_fn)

    stats = evaluate(model, loader, device)
    print(stats)
    print(f"{stats["map_50"]=}, {stats["map"]=}")


if __name__ == "__main__":
    main()
