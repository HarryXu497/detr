from __future__ import annotations

import torch

from detr.detr import DETROutput
from detr.postprocess import postprocess


def test_postprocess_drops_no_object_and_scales_boxes():
    """Each query's class is the top *non*-no-object logit, and boxes are converted from
    normalized cxcywh to absolute xyxy in the original image size."""
    # 1 image, 2 queries, 2 classes + no-object (index 2).
    logits = torch.full((1, 2, 3), -10.0)
    logits[0, 0, 1] = 10.0     # query 0 -> class 1, high score
    logits[0, 1, 2] = 10.0     # query 1 -> no-object dominates
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4],
                           [0.1, 0.1, 0.1, 0.1]]])
    sizes = torch.tensor([[200, 100]])   # (H, W)

    out = postprocess(DETROutput(pred_logits=logits, pred_boxes=boxes), sizes)[0]

    # Top non-object class of query 0 is class 1, with near-full confidence.
    assert out.labels[0].item() == 1
    assert out.scores[0].item() > 0.99
    # Query 1's mass sits on the dropped no-object slot, so its non-object score is ~0.
    assert out.scores[1].item() < 0.01
    # box 0: cx=0.5*100=50, w=0.4*100=40 -> x0=30,x1=70; cy=0.5*200=100, h=80 -> y0=60,y1=140
    assert torch.allclose(out.boxes[0], torch.tensor([30.0, 60.0, 70.0, 140.0]), atol=1e-4)


def test_postprocess_returns_one_output_per_image_with_expected_shapes():
    B, N, num_classes = 3, 5, 4
    logits = torch.randn(B, N, num_classes + 1)
    boxes = torch.rand(B, N, 4)
    sizes = torch.tensor([[100, 200], [640, 480], [50, 50]])

    outs = postprocess(DETROutput(pred_logits=logits, pred_boxes=boxes), sizes)

    assert len(outs) == B
    for o in outs:
        assert o.scores.shape == (N,)
        assert o.labels.shape == (N,)
        assert o.boxes.shape == (N, 4)
        # labels index real classes only (0..num_classes-1); the no-object slot is dropped.
        assert int(o.labels.max()) < num_classes


def test_postprocess_scales_each_image_by_its_own_size():
    """A box at the full-image center scales to that image's own dimensions."""
    boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]], [[0.5, 0.5, 1.0, 1.0]]])
    logits = torch.zeros(2, 1, 2)   # 1 class + no-object
    sizes = torch.tensor([[100, 300], [400, 200]])   # (H, W) per image

    outs = postprocess(DETROutput(pred_logits=logits, pred_boxes=boxes), sizes)

    # full-frame box -> (0, 0, W, H) for each image's own size
    assert torch.allclose(outs[0].boxes[0], torch.tensor([0.0, 0.0, 300.0, 100.0]), atol=1e-4)
    assert torch.allclose(outs[1].boxes[0], torch.tensor([0.0, 0.0, 200.0, 400.0]), atol=1e-4)
