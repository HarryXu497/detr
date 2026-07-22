"""Bounding-box geometry: format conversions and IoU / generalized IoU.

Two box formats are used throughout DETR:

- ``cxcywh``: ``(center_x, center_y, width, height)``, normalized to ``[0, 1]``.
  This is what the model *predicts* (box head + sigmoid).
- ``xyxy``: ``(x0, y0, x1, y1)`` = top-left and bottom-right corners. Required for
  any intersection/area math.

The pairwise functions (:func:`box_iou`, :func:`generalized_box_iou`) take ``(N, 4)``
and ``(M, 4)`` and return an ``(N, M)`` matrix — every box in the first set scored
against every box in the second — which is exactly what the Hungarian matcher needs.
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """Convert ``(..., 4)`` boxes from center format to corner format.

    ``(cx, cy, w, h) -> (x0, y0, x1, y1)`` where ``x0 = cx - w/2`` etc. Works for
    any leading shape.
    """
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """Convert ``(..., 4)`` boxes from corner format to center format.

    ``(x0, y0, x1, y1) -> (cx, cy, w, h)``; the inverse of :func:`box_cxcywh_to_xyxy`.
    """
    x1, y1, x2, y2 = x.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of each ``xyxy`` box. ``(..., 4) -> (...,)``."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1) * (y2 - y1)


def box_iou(b1: Tensor, b2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        b1: ``(N, 4)`` boxes in ``xyxy``.
        b2: ``(M, 4)`` boxes in ``xyxy``.

    Returns:
        ``(iou, union)``, each of shape ``(N, M)``. ``union`` is returned so that
        :func:`generalized_box_iou` can reuse it without recomputing.

    The intersection rectangle uses the elementwise ``max`` of the two top-lefts and
    the elementwise ``min`` of the two bottom-rights; ``clamp(min=0)`` makes
    non-overlapping boxes contribute zero area instead of a spurious positive one.
    Broadcasting a ``(N, 1, 2)`` tensor against a ``(M, 2)`` tensor yields the
    ``(N, M, 2)`` grid of every-pair comparisons.
    """
    # Guard against malformed boxes (bottom-right must be >= top-left).
    assert (b1[:, 2:] >= b1[:, :2]).all()
    assert (b2[:, 2:] >= b2[:, :2]).all()

    area1 = box_area(b1)  # (N,)
    area2 = box_area(b2)  # (M,)

    lt = torch.max(b1[:, None, :2], b2[:, :2])  # (N, M, 2) top-left of intersection
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])  # (N, M, 2) bottom-right of intersection

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)

    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    union = area1[:, None] + area2 - intersection  # (N, M)

    iou = intersection / union
    return iou, union


def generalized_box_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """Pairwise generalized IoU (GIoU) between two sets of ``xyxy`` boxes.

    Args:
        b1: ``(N, 4)`` boxes in ``xyxy``.
        b2: ``(M, 4)`` boxes in ``xyxy``.

    Returns:
        ``(N, M)`` GIoU matrix, valued in ``[-1, 1]``.

    GIoU augments IoU with a penalty for the empty space in the smallest enclosing
    box ``C``::

        giou = iou - (area(C) - union) / area(C)

    Unlike IoU, this stays informative (nonzero gradient) when boxes do not overlap:
    the further apart they are, the larger and emptier ``C`` becomes and the more
    negative GIoU gets, so the loss can still pull a badly-placed box toward its
    target. The enclosing box uses the *opposite* corners of the intersection: the
    ``min`` of the top-lefts and the ``max`` of the bottom-rights.
    """
    iou, union = box_iou(b1, b2)

    lt = torch.min(b1[:, None, :2], b2[:, :2])  # (N, M, 2) top-left of enclosing box
    rb = torch.max(b1[:, None, 2:], b2[:, 2:])  # (N, M, 2) bottom-right of enclosing box

    enclosing_box = (rb - lt).clamp(min=0)  # (N, M, 2)
    enclosing_area = enclosing_box[:, :, 0] * enclosing_box[:, :, 1]  # (N, M)

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou
