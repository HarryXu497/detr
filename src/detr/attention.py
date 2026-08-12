"""Multi-head attention.

Implements scaled dot-product attention over multiple heads::

    Attention(Q, K, V) = softmax(Q Kᵀ / √head_dim) V

The interface takes positional encodings as separate arguments, added to the query
and key but not the value, and re-supplied on each call. An optional key-padding mask
excludes padded positions from the attention.

Tensors use the sequence-first layout ``(L, B, D)`` (length, batch, feature), matching
the default layout of :class:`torch.nn.MultiheadAttention`.
"""
from __future__ import annotations
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Args:
        d_model: model/feature dimension; must be divisible by ``nheads``.
        nheads: number of attention heads, each operating on ``d_model // nheads`` dims.
        dropout: dropout applied to the attention weights.

    The query, key, value, and output projections are separate ``nn.Linear`` layers.
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0):
        super().__init__()
        self._dropout = nn.Dropout(dropout)
        self._nheads = nheads
        self._head_dim, remainder = divmod(d_model, nheads)

        if remainder != 0:
            raise ValueError("'d_model' must be divisible by 'nheads'.")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            query_pos: Tensor | None = None,
            key_pos: Tensor | None = None,
            key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute attention of ``query`` over ``key`` and ``value``.

        Args:
            query: ``(Lq, B, D)`` query sequence.
            key: ``(Lk, B, D)`` key sequence.
            value: ``(Lk, B, D)`` value sequence.
            query_pos: ``(Lq, B, D)`` positional encoding added to the query only.
            key_pos: ``(Lk, B, D)`` positional encoding added to the key only.
            key_padding_mask: ``(B, Lk)`` bool mask; ``True`` marks a key to exclude.

        Returns:
            ``(Lq, B, D)`` output, one vector per query position.
        """
        # Add position encoding to the query and key
        q: Tensor = self.q_proj(query + query_pos if query_pos is not None else query)
        k: Tensor = self.k_proj(key + key_pos if key_pos is not None else key)
        v: Tensor = self.v_proj(value)

        # Split the FEATURE dim into heads i.e. split each query vector into heads
        # Replace the 'D' dimension with (self._nheads, self._head_dim)
        # Then each head attends to sections of EVERY query vector
        Lq, B, D = query.shape
        Lk = key.shape[0]
        # (Lq/Lk, B, nheads, head_dim) -> (B, nheads, Lq/Lk, head_dim)
        q = q.view(Lq, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)
        k = k.view(Lk, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)
        v = v.view(Lk, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)

        # Scaled dot-product scores
        # (B, nheads, Lq, head_dim) @ (B, nheads, Lk, head_dim).transpose(-2, -1)
        # (B, nheads, Lq, head_dim) @ (B, nheads, head_dim, Lk)
        # B and nheads are the batch; Matmul has (Lq, head_dim) @ (head_dim, Lk) -> (Lq, Lk)
        # Result: (B, nheads, Lq, Lk)
        scores: Tensor = q @ k.transpose(-2, -1) / (self._head_dim ** 0.5)

        # Set excluded keys to -inf so their softmax weight is zero.
        if key_padding_mask is not None:
            # (B, 1, 1, Lk) is broadcast to (B, nheads, Lq, Lk)
            # Masks out certain values vectors per batch by setting the corresponding score to -inf
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax and dropout (B, nheads, Lq, Lk)
        # Softmax rescales along Lk dimension
        weights = scores.softmax(dim=-1)
        weights = self._dropout(weights)

        # Weighted sum of value vectors from attention scores
        # (B, nheads, Lq, Lk) @ (B, nheads, Lk, head_dim) = (B, nheads, Lq, head_dim)
        out: Tensor = weights @ v

        # (B, nheads, Lq, head_dim) -> (Lq, B, nheads, head_dim) -> (Lq, B, D)
        out = out.permute(2, 0, 1, 3).reshape(Lq, B, D)

        return self.out_proj(out)
