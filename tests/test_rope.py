# Gate 3 (unit, CPU): RoPE properties. PLAN.md §6 Gate 3.
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.motip.rope import apply_rope, rope_positions, RoPECrossAttention


def test_relative_offset_invariance():
    """<R(p_i) q, R(p_j) k> depends only on p_i - p_j (random q, k, shifted positions)."""
    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 5, 32), torch.randn(1, 4, 6, 32)
    pq, pk = torch.randint(0, 30, (1, 5)).float(), torch.randint(0, 30, (1, 6)).float()
    s0 = apply_rope(q, pq) @ apply_rope(k, pk).transpose(-1, -2)
    for shift in (1.0, 7.0, 29.0):
        s1 = apply_rope(q, pq + shift) @ apply_rope(k, pk + shift).transpose(-1, -2)
        assert torch.allclose(s0, s1, atol=1e-4), f"shift {shift}"
    # ...and it genuinely depends on the offset
    s2 = apply_rope(q, pq + 3.0) @ apply_rope(k, pk).transpose(-1, -2)
    assert not torch.allclose(s0, s2, atol=1e-3)


def test_position_zero_is_identity_and_norm_preserved():
    x = torch.randn(2, 3, 4, 32)
    assert torch.allclose(apply_rope(x, torch.zeros(2, 4)), x)
    y = apply_rope(x, torch.randint(0, 50, (2, 4)).float())
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_positions_alpha_modes_are_inference_only():
    t = torch.tensor([[0, 1, 2, 3]])
    assert torch.equal(rope_positions(t, 2.0, "multiply", training=True), t.float())
    assert torch.equal(rope_positions(t, 2.0, "multiply", training=False), t.float() * 2)
    assert torch.equal(rope_positions(t, 2.0, "divide", training=False), t.float() / 2)
    assert torch.equal(rope_positions(t, 1.0, "divide", training=False), t.float())


def _attn_inputs(B=2, Lq=5, Lk=7, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Lq, D, generator=g), torch.randn(B, Lk, D, generator=g),
            torch.arange(Lq).float().repeat(B, 1), torch.arange(Lk).float().repeat(B, 1))


def test_attention_shape_and_shift_invariance():
    torch.manual_seed(0)
    att = RoPECrossAttention(64, 2).eval()
    q, kv, qp, kp = _attn_inputs()
    allowed = torch.ones(2, 5, 7, dtype=torch.bool)
    o0 = att(q, kv, kv, qp, kp, allowed)
    assert o0.shape == (2, 5, 64)
    o1 = att(q, kv, kv, qp + 11, kp + 11, allowed)
    assert torch.allclose(o0, o1, atol=1e-4)


def test_masked_keys_do_not_influence_and_fully_masked_rows_are_finite():
    torch.manual_seed(0)
    att = RoPECrossAttention(64, 2).eval()
    q, kv, qp, kp = _attn_inputs()
    allowed = torch.ones(2, 5, 7, dtype=torch.bool)
    allowed[:, :, 4:] = False                       # keys 4..6 are "future"/padded
    allowed[:, 2, :] = False                        # query 2 sees nothing at all
    o0 = att(q, kv, kv, qp, kp, allowed)
    kv2 = kv.clone(); kv2[:, 4:] += 100.0           # perturb masked keys only
    o1 = att(q, kv2, kv2, qp, kp, allowed)
    assert torch.allclose(o0, o1, atol=1e-5)
    assert torch.isfinite(o0).all()
    # fully masked row attends only to the zero slot -> output is the out_proj bias (add_zero_attn semantics)
    assert torch.allclose(o0[:, 2], att.out_proj.bias.expand(2, -1), atol=1e-6)
