"""Bounding-box geometry: format conversions and IoU / generalized IoU.

Two box formats are used:

- ``cxcywh``: ``(center_x, center_y, width, height)``, normalized to ``[0, 1]``.
- ``xyxy``: ``(x0, y0, x1, y1)``, the top-left and bottom-right corners.

The pairwise functions (:func:`box_iou`, :func:`generalized_box_iou`) take ``(N, 4)``
and ``(M, 4)`` boxes and return an ``(N, M)`` matrix scoring every box in the first set
against every box in the second.
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """Convert ``(..., 4)`` boxes from ``cxcywh`` to ``xyxy`` format."""
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """Convert ``(..., 4)`` boxes from ``xyxy`` to ``cxcywh`` format.

    Inverse of :func:`box_cxcywh_to_xyxy`.
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
        ``(iou, union)``, each of shape ``(N, M)``. ``union`` is returned for reuse by
        :func:`generalized_box_iou`.
    """
    assert (b1[:, 2:] >= b1[:, :2]).all()
    assert (b2[:, 2:] >= b2[:, :2]).all()

    area1 = box_area(b1)
    area2 = box_area(b2)

    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])

    wh = (rb - lt).clamp(min=0)

    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection

    iou = intersection / union
    return iou, union


def generalized_box_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """Pairwise generalized IoU (GIoU) between two sets of ``xyxy`` boxes.

    Args:
        b1: ``(N, 4)`` boxes in ``xyxy``.
        b2: ``(M, 4)`` boxes in ``xyxy``.

    Returns:
        ``(N, M)`` GIoU matrix, valued in ``[-1, 1]``.

    GIoU augments IoU with a penalty for the empty area of the smallest enclosing
    box ``C``::

        giou = iou - (area(C) - union) / area(C)

    The enclosing box uses the ``min`` of the top-lefts and the ``max`` of the
    bottom-rights.
    """
    iou, union = box_iou(b1, b2)

    lt = torch.min(b1[:, None, :2], b2[:, :2])
    rb = torch.max(b1[:, None, 2:], b2[:, 2:])

    enclosing_box = (rb - lt).clamp(min=0)
    enclosing_area = enclosing_box[:, :, 0] * enclosing_box[:, :, 1]

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou
