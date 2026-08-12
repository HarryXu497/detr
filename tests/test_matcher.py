from __future__ import annotations
import torch
from detr.detr import DETROutput
from detr.matcher import HungarianMatcher


def _sorted_pairs(pred_idx, gt_idx):
    order = torch.argsort(gt_idx)
    return pred_idx[order].tolist(), gt_idx[order].tolist()


def test_recovers_known_assignment():
    """Query 0 matches the gt at (0.5,0.5) class 1; query 2 matches the gt at (0.8,0.8)
    class 0; query 1 is junk. The matcher must pair gt0->q2 and gt1->q0."""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 2, 0] = 10.0
    logits[0, 1, 2] = 10.0
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.8, 0.8, 0.2, 0.2]]])
    targets = [{
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.8, 0.8, 0.2, 0.2],
                               [0.5, 0.5, 0.2, 0.2]]),
    }]
    out = matcher(DETROutput(pred_logits=logits, pred_boxes=boxes), targets)
    assert _sorted_pairs(*out[0]) == ([2, 0], [0, 1])


def test_one_pair_list_per_image_with_right_sizes():
    matcher = HungarianMatcher()
    logits = torch.randn(2, 5, 3)
    boxes = torch.rand(2, 5, 4)
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
        {"labels": torch.tensor([1, 0]), "boxes": torch.rand(2, 4)},
    ]
    out = matcher(DETROutput(pred_logits=logits, pred_boxes=boxes), targets)
    assert len(out) == 2
    assert out[0][0].numel() == 1   # image 0 has 1 gt
    assert out[1][0].numel() == 2   # image 1 has 2 gts


def test_no_gradient_through_matching():
    matcher = HungarianMatcher()
    logits = torch.randn(1, 4, 3, requires_grad=True)
    boxes = torch.rand(1, 4, 4, requires_grad=True)
    targets = [{"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)}]
    out = matcher(DETROutput(pred_logits=logits, pred_boxes=boxes), targets)
    assert not out[0][0].requires_grad and not out[0][1].requires_grad


def test_image_with_no_ground_truths():
    matcher = HungarianMatcher()
    logits = torch.randn(1, 5, 3)
    boxes = torch.rand(1, 5, 4)
    targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
    out = matcher(DETROutput(pred_logits=logits, pred_boxes=boxes), targets)
    assert out[0][0].numel() == 0 and out[0][1].numel() == 0
