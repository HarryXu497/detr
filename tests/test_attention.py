import torch
import torch.nn as nn
from detr.attention import MultiHeadAttention


def test_output_shape():
    mha = MultiHeadAttention(d_model=32, nheads=4)
    q = torch.randn(5, 2, 32)
    k = torch.randn(7, 2, 32)
    v = torch.randn(7, 2, 32)
    out = mha(q, k, v)
    assert out.shape == (5, 2, 32)  # output length matches the query, not the key


def test_matches_torch_mha_oracle():
    """Numeric equivalence with nn.MultiheadAttention when pos is None and weights
    are copied across. Proves the core attention math is correct."""
    torch.manual_seed(0)
    d, h = 32, 4
    mine = MultiHeadAttention(d_model=d, nheads=h, dropout=0.0).eval()
    ref = nn.MultiheadAttention(embed_dim=d, num_heads=h, dropout=0.0, batch_first=False).eval()

    # nn.MultiheadAttention fuses q,k,v into one (3d, d) in_proj; chunk it back apart.
    with torch.no_grad():
        wq, wk, wv = ref.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = ref.in_proj_bias.chunk(3, dim=0)
        mine.q_proj.weight.copy_(wq); mine.q_proj.bias.copy_(bq)
        mine.k_proj.weight.copy_(wk); mine.k_proj.bias.copy_(bk)
        mine.v_proj.weight.copy_(wv); mine.v_proj.bias.copy_(bv)
        mine.out_proj.weight.copy_(ref.out_proj.weight)
        mine.out_proj.bias.copy_(ref.out_proj.bias)

    q = torch.randn(5, 2, d)
    k = torch.randn(7, 2, d)
    v = torch.randn(7, 2, d)
    out_mine = mine(q, k, v)
    out_ref, _ = ref(q, k, v)
    assert torch.allclose(out_mine, out_ref, atol=1e-5)


def test_self_attention_shape():
    """Self-attention: query, key, value are the same sequence."""
    mha = MultiHeadAttention(d_model=32, nheads=4)
    x = torch.randn(6, 3, 32)
    assert mha(x, x, x).shape == (6, 3, 32)


def test_query_pos_added_to_query_not_value():
    """Adding query_pos changes the attention weights (Q changes), so the output
    must differ from the no-pos case."""
    torch.manual_seed(1)
    mha = MultiHeadAttention(d_model=16, nheads=2, dropout=0.0).eval()
    q = torch.randn(3, 1, 16)
    kv = torch.randn(4, 1, 16)
    qpos = torch.randn(3, 1, 16)
    out_with_pos = mha(q, kv, kv, query_pos=qpos)
    out_no_pos = mha(q, kv, kv)
    assert not torch.allclose(out_with_pos, out_no_pos)


def test_key_pos_added_to_key_not_value():
    """Adding key_pos changes the attention weights via K."""
    torch.manual_seed(3)
    mha = MultiHeadAttention(d_model=16, nheads=2, dropout=0.0).eval()
    q = torch.randn(3, 1, 16)
    kv = torch.randn(4, 1, 16)
    kpos = torch.randn(4, 1, 16)
    assert not torch.allclose(mha(q, kv, kv, key_pos=kpos), mha(q, kv, kv))


def test_key_padding_mask_ignores_padded_keys():
    """A fully-masked key position must not affect the output, no matter its value."""
    torch.manual_seed(2)
    mha = MultiHeadAttention(d_model=16, nheads=2, dropout=0.0).eval()
    q = torch.randn(3, 1, 16)
    k = torch.randn(4, 1, 16)
    v = torch.randn(4, 1, 16)
    mask = torch.tensor([[False, False, False, True]])  # last key is padding
    out_masked = mha(q, k, v, key_padding_mask=mask)
    v2 = v.clone(); v2[3] += 100.0  # perturb only the masked key's value
    out_masked2 = mha(q, k, v2, key_padding_mask=mask)
    assert torch.allclose(out_masked, out_masked2, atol=1e-5)
