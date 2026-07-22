"""2D sinusoidal positional encoding for image feature maps.

Produces a fixed (non-learned) encoding that assigns each cell of an ``(h, w)`` grid a
distinct vector, so the permutation-invariant attention can recover spatial position.
Row and column coordinates are each encoded with ``num_pos_feats`` sinusoidal channels
and concatenated, giving ``2 * num_pos_feats`` channels total (256 for ``d_model=256``).
"""

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

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_x, pos_y), dim=3)

        return pos.permute(0, 3, 1, 2)