"""Transformer encoder/decoder layers and the assembled stack.

All tensors use the sequence-first layout ``(L, B, D)``.

- :class:`TransformerEncoderLayer`: self-attention sub-layer followed by a
  feed-forward sub-layer.
- :class:`TransformerDecoderLayer`: query self-attention, cross-attention into the
  encoder memory, and a feed-forward sub-layer.
- :class:`Transformer`: stacks the layers and returns the output of every decoder
  layer to support auxiliary losses.

Positional encodings are supplied on each call and re-added at every layer. The
decoder content stream (``target``) is initialized to zeros; the learned object
queries are supplied separately as ``query_pos``.
"""

import torch
from torch import nn, Tensor

from detr.attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """Encoder layer: self-attention sub-layer followed by a feed-forward sub-layer.

    Each sub-layer applies the post-norm residual form ``norm(x + dropout(sublayer(x)))``.
    """

    def __init__(self, d_model: int = 256, nheads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: Tensor, pos: Tensor,
                src_key_padding_mask: Tensor | None = None) -> Tensor:
        """Apply self-attention and the feed-forward sub-layer.

        Args:
            src: ``(L, B, D)`` input sequence.
            pos: ``(L, B, D)`` positional encoding for ``src``.
            src_key_padding_mask: ``(B, L)`` mask marking padded positions, or ``None``.

        Returns:
            ``(L, B, D)`` output sequence.
        """
        # Self attention with key = value = query = src
        attention_out = self.self_attn(src, src, src, pos, pos, src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attention_out))

        # FFN for non-linearity
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ffn_out))
        return src


class TransformerDecoderLayer(nn.Module):
    """Decoder layer: query self-attention, cross-attention into memory, and FFN.

    ``target`` is the content stream, refined by each layer. ``query_pos`` is the
    learned object-query positional encoding.
    """

    def __init__(self, d_model: int = 256, nheads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
            self,
            target: Tensor,
            memory: Tensor,
            pos: Tensor,
            query_pos: Tensor,
            memory_key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """Refine ``target`` through self-attention, cross-attention, and FFN.

        Args:
            target: ``(N, B, D)`` content stream.
            memory: ``(L, B, D)`` encoder output.
            pos: ``(L, B, D)`` positional encoding for ``memory``.
            query_pos: ``(N, B, D)`` object-query positional encoding.
            memory_key_padding_mask: ``(B, L)`` mask marking padded memory positions,
                or ``None``.

        Returns:
            ``(N, B, D)`` refined content.
        """
        # Self attention with key = value = query = target
        self_attn_out = self.self_attn(target, target, target, query_pos, query_pos)
        target = self.norm1(target + self.dropout1(self_attn_out))

        # Cross attention with key and value as memory/context (should be from the encoder)
        cross_attn_out = self.cross_attn(target, memory, memory, query_pos, pos, memory_key_padding_mask)
        target = self.norm2(target + self.dropout2(cross_attn_out))

        # FFN for non-linearity
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = self.norm3(target + self.dropout3(ffn_out))
        return target


class Transformer(nn.Module):
    """Full transformer: a stack of encoder layers followed by a stack of decoder layers.

    Returns every decoder layer's output stacked along a new leading axis to support
    auxiliary losses.
    """

    def __init__(
        self,
        d_model: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, src: Tensor, pos_embed: Tensor, query_embed: Tensor, mask: Tensor | None = None) -> Tensor:
        """Run the encoder followed by the decoder.

        Args:
            src: ``(L, B, D)`` flattened image tokens.
            pos_embed: ``(L, B, D)`` positional encoding for the image tokens.
            query_embed: ``(N, D)`` object-query embeddings, shared across the batch.
            mask: ``(B, L)`` image-token padding mask, or ``None``.

        Returns:
            ``(num_decoder_layers, N, B, D)`` stacked decoder outputs.
        """
        B = src.shape[1]
        # (N, D) -> (N, 1, D) -> (N, B, D)
        query_pos = query_embed.unsqueeze(dim=1).repeat(1, B, 1)

        # Target is modified by the decoder, starts as (N, B, D) of zeroes
        target = torch.zeros_like(query_pos)

        # Run src through encoder; final output becomes memory/context for decoder
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, pos_embed, mask)

        # Use all decoder outputs for stronger gradient flow
        outputs = []
        for decoder_layer in self.decoder_layers:
            target = decoder_layer(target, memory, pos_embed, query_pos, mask)
            outputs.append(target)

        return torch.stack(outputs)
