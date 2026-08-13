"""Decode DETR predictions into detections.

Turns the model's raw ``pred_logits`` / ``pred_boxes`` into per-image scores, labels, and
absolute-pixel boxes, following DETR's no-NMS rule: every query keeps its best non-object
class. Shared by :mod:`detr.predict` (visualization) and evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

from detr.box_ops import box_cxcywh_to_xyxy
from detr.detr import DETROutput


@dataclass(frozen=True, kw_only=True, slots=True)
class PostProcessOutput:
    """Decoded detections for one image.

    Attributes:
        scores: ``(N,)`` confidence of each query's predicted class.
        labels: ``(N,)`` predicted class index, always a real class (the no-object slot
            is never selected).
        boxes: ``(N, 4)`` boxes in absolute ``xyxy`` pixels of the original image.
    """

    scores: Tensor
    labels: Tensor
    boxes: Tensor


@torch.no_grad()
def postprocess(output: DETROutput, target_sizes: Tensor) -> list[PostProcessOutput]:
    """Decode raw model output into per-image detections.

    For each query, the confidence and label come from the top class *excluding* the
    no-object slot (DETR's no-NMS rule: keep every query's best real class), and the box
    is converted from normalized ``cxcywh`` to absolute ``xyxy`` scaled to the original
    image size. Filtering by score is left to the caller.

    Args:
        output: model predictions for one decoder layer, with ``pred_logits``
            ``(B, N, num_classes + 1)`` and ``pred_boxes`` ``(B, N, 4)`` in normalized
            ``cxcywh``.
        target_sizes: ``(B, 2)`` original image sizes as ``(H, W)`` per image.

    Returns:
        A length-``B`` list of :class:`PostProcessOutput`, one per image, each holding
        ``(N,)`` scores, ``(N,)`` labels, and ``(N, 4)`` absolute ``xyxy`` boxes.
    """
    # doesnt change shape; softmax along class dim (B, N, C + 1)
    prob = output.pred_logits.softmax(-1)
    # Drop the last class (no-object) and max along class dim
    # Each (B, N)
    scores, labels = prob[..., :-1].max(-1)

    # (B, N, 4)
    boxes = box_cxcywh_to_xyxy(output.pred_boxes)

    # (B, 2) to (B, ) each
    img_h, img_w = target_sizes.unbind(-1)
    # (B, 4)
    scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    # (B, N, 4) * (B, 1, 4) = (B, N, 4); broadcast scale[1] to dim N
    boxes = boxes * scale[:, None, :]

    return [
        # scores (N,), labels (N,), boxes (N, 4)
        PostProcessOutput(scores=s, labels=l, boxes=b) for s, l, b in zip(scores, labels, boxes, strict=True)
    ]
