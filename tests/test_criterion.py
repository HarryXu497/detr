import torch

from detr.detr import DETROutput, DETROutputWithAuxOutputs
from detr.matcher import HungarianMatcher
from detr.criterion import SetCriterion


def _make_criterion(num_classes=3, eos_coef=0.1):
    return SetCriterion(
        num_classes=num_classes,
        matcher=HungarianMatcher(),
        weight_dict={},
        eos_coef=eos_coef,
    )


def test_empty_weight_buffer_down_weights_no_object():
    crit = _make_criterion(num_classes=3, eos_coef=0.1)
    assert crit.empty_weight.shape == (4,)
    assert crit.empty_weight[:3].tolist() == [1.0, 1.0, 1.0]
    assert torch.isclose(crit.empty_weight[3], torch.tensor(0.1))
    # Registered as a buffer, so it is module state but not a parameter.
    assert "empty_weight" in dict(crit.named_buffers())
    assert "empty_weight" not in dict(crit.named_parameters())


def test_get_src_permutation_idx_flattens_matches():
    crit = _make_criterion()
    indices = [
        (torch.tensor([5, 12]), torch.tensor([0, 1])),   # image 0: two matches
        (torch.tensor([3]), torch.tensor([0])),           # image 1: one match
    ]
    src_index, batch_index = crit._get_src_permutation_idx(indices)
    assert src_index.tolist() == [5, 12, 3]
    assert batch_index.tolist() == [0, 0, 1]
    assert src_index.dtype == torch.long and batch_index.dtype == torch.long


def test_perfect_prediction_gives_zero_loss():
    """Confident correct classes and exact boxes drive every loss term to ~0."""
    crit = _make_criterion(num_classes=3)
    boxes_gt = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]])
    targets = [{"labels": torch.tensor([0, 1]), "boxes": boxes_gt}]

    logits = torch.full((1, 5, 4), -10.0)
    logits[0, 0, 0] = 10.0   # query 0 -> class 0
    logits[0, 2, 1] = 10.0   # query 2 -> class 1
    logits[0, 1, 3] = logits[0, 3, 3] = logits[0, 4, 3] = 10.0   # rest -> no-object
    boxes = torch.rand(1, 5, 4) * 0.1
    boxes[0, 0] = boxes_gt[0]
    boxes[0, 2] = boxes_gt[1]

    out = DETROutputWithAuxOutputs(
        output=DETROutput(pred_logits=logits, pred_boxes=boxes), aux_outputs=[]
    )
    loss = crit(out, targets)
    assert loss.loss_ce.item() < 1e-3
    assert loss.loss_box.loss_bbox.item() < 1e-6
    assert loss.loss_box.loss_giou.item() < 1e-6


def test_loss_boxes_l1_normalized_by_num_boxes():
    """L1 is summed over coordinates and divided by num_boxes, giving a per-object mean."""
    crit = _make_criterion()
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]])
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])}]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    out = DETROutput(pred_logits=torch.empty(1, 2, 4), pred_boxes=pred_boxes)

    # Each pair contributes |0.2-0.4| + |0.2-0.4| = 0.4 in L1; two pairs / num_boxes=2.
    box_loss = crit.loss_boxes(out, targets, indices, num_boxes=2)
    assert torch.isclose(box_loss.loss_bbox, torch.tensor(0.4), atol=1e-6)


def test_loss_giou_uses_matched_pairs_not_full_matrix():
    """With each prediction equal to its own target, only the diagonal GIoU is 1, so a
    correct implementation yields ~0 loss; summing the full matrix would not."""
    crit = _make_criterion()
    a = [0.3, 0.3, 0.2, 0.2]
    b = [0.7, 0.7, 0.2, 0.2]
    pred_boxes = torch.tensor([[a, b]])
    targets = [{"boxes": torch.tensor([a, b])}]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    out = DETROutput(pred_logits=torch.empty(1, 2, 4), pred_boxes=pred_boxes)

    box_loss = crit.loss_boxes(out, targets, indices, num_boxes=2)
    assert box_loss.loss_giou.item() < 1e-6


def test_eos_coef_shifts_classification_loss():
    """Down-weighting the no-object class lowers the loss contributed by wrong,
    unmatched queries, so a smaller eos_coef yields a smaller loss here."""
    targets = [{"labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 0, 0] = 10.0    # matched query 0 -> class 0 (correct)
    logits[0, 1:, 0] = 10.0   # unmatched queries confidently predict class 0 (wrong)
    out = DETROutput(pred_logits=logits, pred_boxes=torch.empty(1, 4, 4))

    low = _make_criterion(eos_coef=0.1).loss_labels(out, targets, indices)
    high = _make_criterion(eos_coef=1.0).loss_labels(out, targets, indices)
    assert low.item() < high.item()


def test_aux_losses_one_entry_per_auxiliary_layer():
    crit = _make_criterion(num_classes=3)
    targets = [{"labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])}]
    main = DETROutput(pred_logits=torch.randn(1, 6, 4), pred_boxes=torch.rand(1, 6, 4))
    aux = [DETROutput(pred_logits=torch.randn(1, 6, 4), pred_boxes=torch.rand(1, 6, 4))
           for _ in range(5)]
    loss = crit(DETROutputWithAuxOutputs(output=main, aux_outputs=aux), targets)
    assert len(loss.loss_ce_aux) == 5
    assert len(loss.loss_box_aux) == 5


def test_image_with_no_ground_truths_is_finite():
    """num_boxes is floored at 1, so an image with no objects yields finite losses."""
    crit = _make_criterion(num_classes=3)
    targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
    out = DETROutputWithAuxOutputs(
        output=DETROutput(pred_logits=torch.randn(1, 5, 4), pred_boxes=torch.rand(1, 5, 4)),
        aux_outputs=[],
    )
    loss = crit(out, targets)
    assert torch.isfinite(loss.loss_ce)
    assert torch.isfinite(loss.loss_box.loss_bbox)
    assert torch.isfinite(loss.loss_box.loss_giou)


def test_gradients_flow_to_predictions():
    crit = _make_criterion(num_classes=3)
    targets = [{"labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])}]
    logits = torch.randn(1, 6, 4, requires_grad=True)
    boxes = torch.rand(1, 6, 4, requires_grad=True)
    out = DETROutputWithAuxOutputs(
        output=DETROutput(pred_logits=logits, pred_boxes=boxes), aux_outputs=[]
    )
    loss = crit(out, targets)
    (loss.loss_ce + loss.loss_box.loss_bbox + loss.loss_box.loss_giou).backward()
    assert logits.grad is not None
    assert boxes.grad is not None
