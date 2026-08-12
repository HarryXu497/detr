from __future__ import annotations
import torch
from detr.position_encoding import PositionEmbeddingSine


def test_output_shape_matches_d_model():
    pe = PositionEmbeddingSine(num_pos_feats=128)
    mask = torch.zeros(2, 5, 7, dtype=torch.bool)  # no padding
    out = pe(mask)
    assert out.shape == (2, 256, 5, 7)  # 2 * 128 = d_model


def test_values_bounded():
    pe = PositionEmbeddingSine(num_pos_feats=128)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = pe(mask)
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_distinct_positions_differ():
    pe = PositionEmbeddingSine(num_pos_feats=128)
    out = pe(torch.zeros(1, 4, 4, dtype=torch.bool))[0]
    assert not torch.allclose(out[:, 0, 0], out[:, 3, 3])


def test_adjacent_cells_differ():
    pe = PositionEmbeddingSine(num_pos_feats=128)
    out = pe(torch.zeros(1, 4, 4, dtype=torch.bool))[0]
    assert not torch.allclose(out[:, 0, 0], out[:, 0, 1])  # differ along x
    assert not torch.allclose(out[:, 0, 0], out[:, 1, 0])  # differ along y


def test_num_pos_feats_controls_channels():
    pe = PositionEmbeddingSine(num_pos_feats=64)
    out = pe(torch.zeros(1, 3, 3, dtype=torch.bool))
    assert out.shape == (1, 128, 3, 3)


def test_normalize_makes_last_cell_size_invariant():
    """With normalize=True the coordinates are rescaled so the last cell always maps to
    2*pi, giving the same encoding regardless of feature-map size."""
    pe = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    small = pe(torch.zeros(1, 4, 4, dtype=torch.bool))[0, :, -1, -1]
    large = pe(torch.zeros(1, 8, 8, dtype=torch.bool))[0, :, -1, -1]
    assert torch.allclose(small, large, atol=1e-5)
