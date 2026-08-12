from __future__ import annotations
from typing import cast

import torch
from detr.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Transformer,
)


def test_encoder_layer_shape():
    layer = TransformerEncoderLayer(d_model=32, nheads=4, dim_feedforward=64, dropout=0.0)
    src = torch.randn(20, 2, 32)
    pos = torch.randn(20, 2, 32)
    assert layer(src, pos).shape == (20, 2, 32)


def test_decoder_layer_shape():
    layer = TransformerDecoderLayer(d_model=32, nheads=4, dim_feedforward=64, dropout=0.0)
    tgt = torch.zeros(5, 2, 32)
    memory = torch.randn(20, 2, 32)
    pos = torch.randn(20, 2, 32)
    query_pos = torch.randn(5, 2, 32)
    # decoder output length matches the queries (5), not the memory (20)
    assert layer(tgt, memory, pos, query_pos).shape == (5, 2, 32)


def test_transformer_returns_stacked_decoder_layers():
    d, N, ndec = 32, 5, 6
    model = Transformer(d_model=d, nheads=4, num_encoder_layers=2,
                        num_decoder_layers=ndec, dim_feedforward=64, dropout=0.0)
    src = torch.randn(20, 2, d)          # L=20, B=2
    pos = torch.randn(20, 2, d)
    query_embed = torch.randn(N, d)      # N queries
    out = model(src, pos, query_embed)
    assert out.shape == (ndec, N, 2, d)  # every decoder layer stacked for aux loss


def test_query_embed_expands_across_batch():
    """The same N queries are broadcast to every image in the batch."""
    d, N = 16, 4
    model = Transformer(d_model=d, nheads=2, num_encoder_layers=1,
                        num_decoder_layers=1, dim_feedforward=32, dropout=0.0)
    src = torch.randn(10, 3, d)          # batch of 3
    pos = torch.randn(10, 3, d)
    query_embed = torch.randn(N, d)
    out = model(src, pos, query_embed)
    assert out.shape == (1, N, 3, d)


def test_decoder_layers_are_independent():
    """The ModuleList comprehension must build distinct layers, not reuse one."""
    ndec = 4
    model = Transformer(d_model=16, nheads=2, num_encoder_layers=1,
                        num_decoder_layers=ndec, dim_feedforward=32, dropout=0.0)
    layers = [cast(TransformerDecoderLayer, layer) for layer in model.decoder_layers]
    weight_ids = {id(layer.self_attn.q_proj.weight) for layer in layers}
    assert len(weight_ids) == ndec


def test_gradients_reach_every_decoder_layer():
    """Backprop through the stacked output must touch all decoder layers (this is what
    makes auxiliary losses on every layer work)."""
    ndec = 3
    model = Transformer(d_model=16, nheads=2, num_encoder_layers=1,
                        num_decoder_layers=ndec, dim_feedforward=32, dropout=0.0)
    out = model(torch.randn(8, 2, 16), torch.randn(8, 2, 16), torch.randn(5, 16))
    out.sum().backward()
    for layer in model.decoder_layers:
        grad = cast(TransformerDecoderLayer, layer).linear1.weight.grad
        assert grad is not None and grad.abs().sum() > 0
