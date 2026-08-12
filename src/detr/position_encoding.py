"""2D sinusoidal positional encoding for image feature maps.

Produces a fixed (non-learned) encoding that assigns each cell of an ``(h, w)`` grid a
distinct vector, so the permutation-invariant attention can recover spatial position.
Row and column coordinates are each encoded with ``num_pos_feats`` sinusoidal channels
and concatenated, giving ``2 * num_pos_feats`` channels total (256 for ``d_model=256``).
"""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D positional encoding.

    Args:
        num_pos_feats: sinusoidal channels per axis; total channels are twice this.
        temperature: base of the geometric frequency progression.
        normalize: if True, scale the row/column coordinates to ``[0, 2*pi]`` so the
            encoding is independent of feature-map size.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0,
                 normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, mask: Tensor) -> Tensor:
        """Compute the positional encoding.

        Args:
            mask: ``(B, h, w)`` bool tensor; ``True`` marks a padded cell.

        Returns:
            ``(B, 2*num_pos_feats, h, w)`` positional encoding.
        """
        not_mask = ~mask

        # Cumsum creates running sum along the dimension; does not collapse the dimension
        # Generates coordinates, not incrementing on masked out cells
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            # The last element is the total sum along the dimension bc of the cumsum
            # Thus this normalizes to [0, 1] * 2pi = [0, 2pi]
            # Divide by (B, h, w) -> (B, h[-1] (1), w)
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            # Divide by (B, h, w) -> (B, h, w[-1] (1))
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        # Compute the denom of the argument to sin and cos
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Compute the argument to sin and cos
        # (B, h, w, 1) / (num_pos_feats,) -> (B, h, w, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        # (B, h, w, 1) / (num_pos_feats,) -> (B, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        # Sin and cos give [(B, h, w, num_pos_feats / 2), (B, h, w, num_pos_feats / 2)] from 0::2 and 1::2
        # Stack gives [(B, h, w, num_pos_feats / 2), (B, h, w, num_pos_feats / 2)]
        # -> (B, h, w, num_pos_feats / 2, 2)
        # Flatten flattens dim 3 onwards to yield (B, h, w, num_pos_feats)
        # Flatten goes in row-major order so it interleaves sin and cos 
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # Cat concatenates (B, h, w, num_pos_feats) and (B, h, w, num_pos_feats)
        # to give (B, h, w, 2*num_pos_feats)
        pos = torch.cat((pos_x, pos_y), dim=3)

        # Permute to (B, h, w, 2*num_pos_feats) -> (B, 2*num_pos_feats, h, w)
        return pos.permute(0, 3, 1, 2)