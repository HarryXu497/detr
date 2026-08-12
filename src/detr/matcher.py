"""Hungarian matcher: assigns predictions to ground-truth boxes one-to-one.

For each image, builds a cost matrix over (prediction, ground-truth) pairs from a
weighted sum of a classification term, an L1 box term, and a GIoU term, then solves the
assignment problem with :func:`scipy.optimize.linear_sum_assignment`. The matching is
non-differentiable and runs under ``torch.no_grad``.
"""
from __future__ import annotations
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
        device = pred_logits.device

        # Flatten predictions across the batch
        B, N = pred_logits.shape[:2]
        # (B * N, C + 1) -> softmaxed
        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        # (B * N, 4)
        out_box = pred_boxes.flatten(0, 1)

        # Concatenate all ground-truths
        # (G, )
        target_ids = torch.cat([t["labels"] for t in targets])
        # (G, 4)
        target_boxes = torch.cat([t["boxes"] for t in targets])

        # Takes only the out_prob values corresponding to ground truth labels
        # (B * N, G)
        cost_class = -out_prob[:, target_ids]

        # Compute pairwise L1 distance between ground-truth and predicted
        # (B * N, G)
        cost_box = torch.cdist(out_box, target_boxes, p=1)

        # Compute pairwise GIoU loss
        # (B * N, G)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_box), box_cxcywh_to_xyxy(target_boxes)
        )

        # Compute weighted sum for total cost
        cost_total = (self.cost_bbox * cost_box + self.cost_class * cost_class + self.cost_giou * cost_giou)
        # Unflatten (B * N, G) to (B, N, G), then run on cpu because of SciPy
        cost_total = cost_total.view(B, N, -1).cpu()

        # Count number of boxes per target image
        sizes = [len(t["boxes"]) for t in targets]
        indices = []

        # Iterate though chunks of costs; each chunk corresponds to one image
        # (B, N, G) -> B items of (B, N, n_i)
        for i, c in enumerate(cost_total.split(sizes, dim=2)):
            # Given (N, n_i) cost matrix, compute the minimum-total-cost
            # assignment of the n_i gts to n_i distinct predictions
            # c[i] is (N, n_i)
            row, col = linear_sum_assignment(c[i])
            indices.append((
                torch.as_tensor(row, dtype=torch.long, device=device),
                torch.as_tensor(col, dtype=torch.long, device=device),
            ))

        return indices
