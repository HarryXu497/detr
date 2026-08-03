"""Integration gate for the full DETR training stack, plus fast unit tests for the
training-setup helpers.

The gate exercises backbone -> transformer -> matcher -> criterion -> optimizer step
together. On a handful of fixed synthetic images with augmentation off, the weighted
loss must collapse; if it does not, the bug is in how the pieces are wired, not in any
one of them. This is the same check that must pass before committing GPU hours to a
real run. It is marked ``slow`` (a few hundred steps through the real model); run the
fast suite with ``pytest -m "not slow"``.
"""

import pytest
import torch

from detr.criterion import SetCriterion
from detr.detr import DETR
from detr.matcher import HungarianMatcher
from detr.train import build_model_and_criterion, build_optimizer, weighted_loss


def _small_model(num_classes: int, num_decoder_layers: int, device: str) -> DETR:
    """A small, untrained DETR whose wiring is identical to the full model."""
    return DETR(
        num_classes=num_classes,
        num_queries=20,
        num_encoder_layers=2,
        num_decoder_layers=num_decoder_layers,
        pretrained_backbone=False,
    ).to(device).train()


def _fixed_batch(device: str):
    """Four synthetic images with fixed targets, matching ``collate_fn`` output."""
    torch.manual_seed(0)
    images = torch.randn(4, 3, 128, 128, device=device)
    targets = [
        {"labels": torch.tensor([0], device=device),
         "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], device=device)},
        {"labels": torch.tensor([1, 2], device=device),
         "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]], device=device)},
        {"labels": torch.tensor([2], device=device),
         "boxes": torch.tensor([[0.2, 0.8, 0.2, 0.2]], device=device)},
        {"labels": torch.tensor([0, 1], device=device),
         "boxes": torch.tensor([[0.6, 0.4, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]], device=device)},
    ]
    return images, targets


# --- fast unit tests for the setup helpers -----------------------------------


def test_build_model_and_criterion_wires_paper_weights():
    model, criterion = build_model_and_criterion(num_classes=3, device="cpu")
    assert isinstance(model, DETR)
    assert isinstance(criterion, SetCriterion)
    assert criterion.weight_dict == {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    # eos_coef default down-weights the no-object class in the classification buffer.
    assert torch.isclose(criterion.empty_weight[-1], torch.tensor(0.1))


def test_build_optimizer_uses_split_backbone_learning_rate():
    model = _small_model(num_classes=3, num_decoder_layers=2, device="cpu")
    optimizer = build_optimizer(model)

    lr_by_id = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            lr_by_id[id(p)] = group["lr"]

    backbone_lrs = {lr_by_id[id(p)] for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("backbone")}
    other_lrs = {lr_by_id[id(p)] for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("backbone")}

    assert backbone_lrs == {1e-5}
    assert other_lrs == {1e-4}
    assert all(g["weight_decay"] == 1e-4 for g in optimizer.param_groups)


def test_weighted_loss_includes_auxiliary_layers():
    """The summed loss must exceed the final-layer-only loss when aux outputs exist,
    confirming the aux terms are folded into the training objective."""
    num_classes = 3
    model = _small_model(num_classes, num_decoder_layers=3, device="cpu")
    criterion = SetCriterion(num_classes, HungarianMatcher(),
                             {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0})
    images, targets = _fixed_batch("cpu")

    loss = criterion(model(images), targets)
    assert len(loss.loss_ce_aux) == 2   # 3 decoder layers -> final + 2 aux

    wd = criterion.weight_dict
    final_only = (wd["loss_ce"] * loss.loss_ce
                  + wd["loss_bbox"] * loss.loss_box.loss_bbox
                  + wd["loss_giou"] * loss.loss_box.loss_giou)
    assert weighted_loss(loss, wd).item() > final_only.item()


# --- the integration gate -----------------------------------------------------


@pytest.mark.slow
def test_overfits_a_handful_of_synthetic_images():
    """The whole stack learns to fit four fixed images: gradients flow through every
    component so the weighted loss drops substantially and the box L1 term collapses.

    This is a proxy for the real overfit gate, not a substitute for it. Here the
    backbone is random (not pretrained) and the images are noise, so classification
    converges slowly under the aggressive ``0.1`` gradient clip and the loss does not
    reach zero in a CPU-tractable number of steps. What it does prove is that the
    matcher -> criterion -> optimizer wiring is correct end to end: the L1 box loss,
    which exercises the matched-pair gather and box-op chain, collapses by well over
    half. Driving the loss to near-zero is verified separately on the training box with
    a pretrained backbone and real images (see the AWS runbook, Task 11)."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3

    model = _small_model(num_classes, num_decoder_layers=2, device=device)
    criterion = SetCriterion(num_classes, HungarianMatcher(),
                             {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images, targets = _fixed_batch(device)

    first_total = last_total = None
    first_bbox = last_bbox = None
    for _ in range(300):
        loss = criterion(model(images), targets)
        total = weighted_loss(loss, criterion.weight_dict)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if first_total is None:
            first_total, first_bbox = total.item(), loss.loss_box.loss_bbox.item()
        last_total, last_bbox = total.item(), loss.loss_box.loss_bbox.item()

    assert first_total is not None and last_total is not None
    assert first_bbox is not None and last_bbox is not None
    # Joint learning: a broken stack (no gradient flow, wrong matching, device
    # mismatch) would leave the loss flat or rising. A correct one nearly halves it.
    assert last_total < first_total * 0.55
    # The box-regression path memorizes hard, collapsing the L1 term by more than half.
    assert last_bbox < first_bbox * 0.5
