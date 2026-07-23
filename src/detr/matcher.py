"""Hungarian matcher: assigns predictions to ground-truth boxes one-to-one.

For each image, builds a cost matrix over (prediction, ground-truth) pairs from a
weighted sum of a classification term, an L1 box term, and a GIoU term, then solves the
assignment problem with :func:`scipy.optimize.linear_sum_assignment`. The matching is
non-differentiable and runs under ``torch.no_grad``.
"""

import torch
from torch import nn, Tensor
from scipy.optimize import linear_sum_assignment

from detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detr.detr import DETROutput


class HungarianMatcher(nn.Module):
    """Bipartite matcher between predictions and ground-truth objects.

    Args:
        cost_class: weight on the classification cost.
        cost_bbox: weight on the L1 box cost.
        cost_giou: weight on the GIoU cost.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, output: DETROutput, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        """Match predictions to ground truths per image.

        Args:
            output: model predictions with ``pred_logits`` ``(B, N, C+1)`` and
                ``pred_boxes`` ``(B, N, 4)`` in ``cxcywh``.
            targets: length-``B`` list of dicts with ``labels`` ``(n,)`` and ``boxes``
                ``(n, 4)`` in ``cxcywh``.

        Returns:
            Length-``B`` list of ``(prediction_indices, gt_indices)`` long tensors.
        """
        pred_logits = output.pred_logits
        pred_boxes = output.pred_boxes

        B, N = pred_logits.shape[:2]
        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_box = pred_boxes.flatten(0, 1)

        target_ids = torch.cat([t["labels"] for t in targets])
        target_boxes = torch.cat([t["boxes"] for t in targets])

        cost_class = -out_prob[:, target_ids]
        cost_box = torch.cdist(out_box, target_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_box), box_cxcywh_to_xyxy(target_boxes)
        )

        cost_total = (self.cost_bbox * cost_box + self.cost_class * cost_class + self.cost_giou * cost_giou)
        cost_total = cost_total.view(B, N, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = []

        for i, c in enumerate(cost_total.split(sizes, dim=2)):
            row, col = linear_sum_assignment(c[i])
            indices.append((
                torch.as_tensor(row, dtype=torch.long),
                torch.as_tensor(col, dtype=torch.long),
            ))

        return indices
