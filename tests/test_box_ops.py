from __future__ import annotations
import torch
from detr.box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
)


def test_cxcywh_roundtrip():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.2, 0.1, 0.1]])
    assert torch.allclose(box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes)), boxes, atol=1e-6)


def test_cxcywh_to_xyxy_known():
    # center (0.5,0.5), size (0.2,0.4) -> x0=0.4,y0=0.3,x1=0.6,y1=0.7
    out = box_cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.2, 0.4]]))
    assert torch.allclose(out, torch.tensor([[0.4, 0.3, 0.6, 0.7]]), atol=1e-6)


def test_conversions_preserve_leading_shape():
    # Conversions must work for any (..., 4) shape, e.g. batched (B, N, 4).
    boxes = torch.rand(3, 5, 4)
    assert box_cxcywh_to_xyxy(boxes).shape == (3, 5, 4)
    assert box_xyxy_to_cxcywh(boxes).shape == (3, 5, 4)


def test_iou_identical_boxes():
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    iou, union = box_iou(b, b)
    assert torch.allclose(iou, torch.tensor([[1.0]]))
    assert torch.allclose(union, torch.tensor([[4.0]]))


def test_iou_half_overlap():
    # [0,0,2,2] and [1,0,3,2] overlap = 2, union = 4+4-2 = 6 -> iou = 1/3
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
    iou, _ = box_iou(a, b)
    assert torch.allclose(iou, torch.tensor([[1.0 / 3.0]]), atol=1e-6)


def test_iou_disjoint_is_zero():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    iou, _ = box_iou(a, b)
    assert torch.allclose(iou, torch.tensor([[0.0]]))


def test_iou_returns_pairwise_matrix():
    # 2 boxes vs 3 boxes -> (2, 3) matrix.
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    iou, union = box_iou(a, b)
    assert iou.shape == (2, 3)
    assert union.shape == (2, 3)
    # box a[0] is identical to b[0] -> iou 1.0; a[0] disjoint from b[2] -> iou 0.0
    assert torch.allclose(iou[0, 0], torch.tensor(1.0))
    assert torch.allclose(iou[0, 2], torch.tensor(0.0))


def test_giou_disjoint_is_less_than_iou():
    # disjoint boxes: iou = 0, giou < 0 (penalized by enclosing box)
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    giou = generalized_box_iou(a, b)
    # enclosing box = [0,0,3,3] area 9; union = 2; giou = 0 - (9-2)/9 = -7/9
    assert torch.allclose(giou, torch.tensor([[-7.0 / 9.0]]), atol=1e-6)


def test_giou_identical_is_one():
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    assert torch.allclose(generalized_box_iou(b, b), torch.tensor([[1.0]]), atol=1e-6)


def test_giou_bounded_in_minus_one_to_one():
    torch.manual_seed(0)
    # Build valid xyxy boxes: top-left + positive size.
    def rand_xyxy(n):
        tl = torch.rand(n, 2)
        wh = torch.rand(n, 2) + 0.1
        return torch.cat([tl, tl + wh], dim=1)

    giou = generalized_box_iou(rand_xyxy(6), rand_xyxy(4))
    assert giou.shape == (6, 4)
    assert (giou <= 1.0 + 1e-6).all() and (giou >= -1.0 - 1e-6).all()
