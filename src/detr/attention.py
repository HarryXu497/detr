"""Hand-rolled multi-head attention (the core of the transformer).

This is the single module every encoder/decoder layer is built from. It implements
scaled dot-product attention over multiple heads::

    Attention(Q, K, V) = softmax(Q Kᵀ / √head_dim) V

with two DETR-specific features baked into the interface:

- **Positional encodings are added to the query and key but never the value**, and
  are passed in separately on every call (rather than pre-added by the caller). This
  lets position steer *who attends to whom* while the values passed along stay raw.
- **A key-padding mask** can zero out padded image pixels so they contribute nothing
  to the attention (used once variable-size images are batched).

Tensors are **sequence-first** ``(L, B, D)`` — length, batch, feature — matching the
default layout of :class:`torch.nn.MultiheadAttention`, which lets the oracle test
compare the two directly.
"""

from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Args:
        d_model: model/feature dimension (must be divisible by ``nheads``).
        nheads: number of attention heads; each operates on ``d_model // nheads`` dims.
        dropout: dropout applied to the attention weights.

    The four projections are kept as separate ``nn.Linear`` layers (rather than one
    fused QKV projection) so their weights map cleanly onto PyTorch's reference module
    in the oracle test.
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._dropout = nn.Dropout(dropout)
        self._nheads = nheads
        self._head_dim = d_model // nheads

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            query_pos: Tensor | None = None,
            key_pos: Tensor | None = None,
            key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute attention of ``query`` over ``key``/``value``.

        Args:
            query: ``(Lq, B, D)`` query sequence.
            key: ``(Lk, B, D)`` key sequence.
            value: ``(Lk, B, D)`` value sequence.
            query_pos: ``(Lq, B, D)`` positional encoding added to the query only.
            key_pos: ``(Lk, B, D)`` positional encoding added to the key only.
            key_padding_mask: ``(B, Lk)`` bool; ``True`` marks a key to ignore.

        Returns:
            ``(Lq, B, D)`` — one output per query, so the output length matches the
            query (this is what makes cross-attention shrink image tokens down to the
            N object queries).
        """
        # Add positions to Q and K (not V), then project. `value` stays raw content.
        q: Tensor = self.q_proj(query + query_pos if query_pos is not None else query)
        k: Tensor = self.k_proj(key + key_pos if key_pos is not None else key)
        v: Tensor = self.v_proj(value)

        # Split the feature dim into heads and move heads next to batch, so a single
        # batched matmul runs every head of every batch item at once:
        #   (L, B, D) -> (L, B, h, head_dim) -> (B, h, L, head_dim)
        Lq, B, D = query.shape
        Lk = key.shape[0]
        q = q.view(Lq, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)
        k = k.view(Lk, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)
        v = v.view(Lk, B, self._nheads, self._head_dim).permute(1, 2, 0, 3)

        # Similarity scores, scaled by √head_dim to keep softmax gradients healthy.
        scores: Tensor = q @ k.transpose(-2, -1) / (self._head_dim ** 0.5)  # (B, h, Lq, Lk)

        # Padded keys get -inf so their softmax weight becomes exactly zero.
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # broadcast over heads and queries
            scores = scores.masked_fill(mask, float("-inf"))

        # Normalize over the keys (last dim), then blend the values.
        weights = scores.softmax(dim=-1)
        weights = self._dropout(weights)

        out: Tensor = weights @ v  # (B, h, Lq, head_dim)

        # Merge heads back to (Lq, B, D). reshape (not view) because permute broke
        # contiguity.
        out = out.permute(2, 0, 1, 3).reshape(Lq, B, D)

        # out_proj mixes information back across the (so-far independent) heads.
        return self.out_proj(out)
