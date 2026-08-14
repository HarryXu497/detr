from __future__ import annotations

import torch
from torch import nn

from detr.detr import DETROutput, DETROutputWithAuxOutputs
from detr.eval import evaluate


class _StubModel(nn.Module):
    """Returns a fixed output regardless of input, so a test can control predictions."""

    def __init__(self, output: DETROutputWithAuxOutputs):
        super().__init__()
        self._output = output

    def forward(self, _images: torch.Tensor, _mask: torch.Tensor | None = None) -> DETROutputWithAuxOutputs:
        return self._output


def _output_for(boxes_cxcywh: torch.Tensor, labels: torch.Tensor, num_classes: int) -> DETROutputWithAuxOutputs:
    """Build a model output whose queries confidently predict ``labels`` at ``boxes``."""
    n = labels.shape[0]
    logits = torch.full((1, n, num_classes + 1), -10.0)
    for i, c in enumerate(labels):
        logits[0, i, int(c)] = 10.0
    return DETROutputWithAuxOutputs(
        output=DETROutput(pred_logits=logits, pred_boxes=boxes_cxcywh[None]), aux_outputs=[]
    )


def test_evaluate_perfect_predictions_gives_map_one():
    """Predictions exactly matching the targets score mAP ~1.0 — this also verifies the
    coordinate spaces align (predicted xyxy vs ground-truth cxcywh->xyxy, both normalized)."""
    gt_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]])
    gt_labels = torch.tensor([0, 1])
    targets = [{"labels": gt_labels, "boxes": gt_boxes}]

    model = _StubModel(_output_for(gt_boxes, gt_labels, num_classes=3))
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    stats = evaluate(model, [(torch.randn(1, 3, 64, 64), mask, targets)], "cpu")  # type: ignore[arg-type]

    assert stats["map_50"].item() > 0.99
    assert stats["map"].item() > 0.99


def test_evaluate_mislocated_predictions_give_zero_map():
    """Confident predictions of the right classes but the wrong locations score ~0, so the
    metric is genuinely discriminating, not trivially returning 1."""
    gt_boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]])
    gt_labels = torch.tensor([0, 1])
    targets = [{"labels": gt_labels, "boxes": gt_boxes}]

    # same labels, but boxes in far corners -> IoU 0 with the targets
    pred_boxes = torch.tensor([[0.9, 0.9, 0.05, 0.05], [0.95, 0.05, 0.05, 0.05]])
    model = _StubModel(_output_for(pred_boxes, gt_labels, num_classes=3))
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    stats = evaluate(model, [(torch.randn(1, 3, 64, 64), mask, targets)], "cpu")  # type: ignore[arg-type]

    assert stats["map_50"].item() < 0.01
