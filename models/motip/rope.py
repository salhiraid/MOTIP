# MOTIP2 Sec. 3.1: "we replace MOTIP's learned discrete temporal bias with rotary position embeddings (RoPE)
# applied inside the query-key product, so attention between two tokens varies with their frame offset."
# "... at inference we rescale the RoPE timescale by a factor alpha to match the average training timescale."
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_positions(times: torch.Tensor, alpha: float = 1.0, mode: str = "multiply", training: bool = True):
    """Positions fed to RoPE. `times` are frame indices inside the context window (clip index in training).

    Training: positions = times (alpha is inference-only).
    Inference (PLAN decision #8): `multiply` -> times * alpha (skill default); `divide` -> times / alpha, i.e. the
    timescale (period) is multiplied by alpha. Which one matches Tab. 6 is decided by the Gate-3 A/B.
    """
    t = times.to(torch.float32)
    if training or alpha == 1.0:
        return t
    if mode == "multiply":
        return t * alpha
    if mode == "divide":
        return t / alpha
    raise ValueError(f"Unknown ROPE_ALPHA_MODE {mode}")


def rope_angles(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """positions (...,) -> angles (..., head_dim/2)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dimension"
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    return positions.to(torch.float32)[..., None] * inv_freq


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate the head dimension of x (B, H, L, hd) by position-dependent angles; positions (B, L).

    RoPE property used by Gate 3: <R(p_i) q, R(p_j) k> depends only on p_i - p_j.
    """
    hd = x.shape[-1]
    ang = rope_angles(positions, hd, base)[:, None, :, :]          # (B, 1, L, hd/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPECrossAttention(nn.Module):
    """Multi-head attention with RoPE on q/k. Parameter layout mirrors nn.MultiheadAttention
    (in_proj_weight / in_proj_bias / out_proj) and `add_zero_attn=True` semantics (an all-zero key/value slot that
    every query may attend to, so fully masked rows stay finite) as used by MOTIP's decoder."""

    def __init__(self, embed_dim: int, num_heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.rope_base = rope_base
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, q_pos, k_pos, allowed):
        """query (B,Lq,D), key/value (B,Lk,D), q_pos (B,Lq), k_pos (B,Lk), allowed (B,Lq,Lk) bool (True = may attend)."""
        B, Lq, D = query.shape
        Lk = key.shape[1]
        H, hd = self.num_heads, self.head_dim
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, w_q, b_q).view(B, Lq, H, hd).transpose(1, 2)
        k = F.linear(key, w_k, b_k).view(B, Lk, H, hd).transpose(1, 2)
        v = F.linear(value, w_v, b_v).view(B, Lk, H, hd).transpose(1, 2)
        # MOTIP2 Sec. 3.1: rotate queries and keys by their frame index (relative-offset-only dependence).
        q = apply_rope(q, q_pos, self.rope_base)
        k = apply_rope(k, k_pos, self.rope_base)
        # add_zero_attn: one extra all-zero key/value that is always attendable.
        k = torch.cat([k, k.new_zeros(B, H, 1, hd)], dim=2)
        v = torch.cat([v, v.new_zeros(B, H, 1, hd)], dim=2)
        allowed = torch.cat([allowed, allowed.new_ones(B, Lq, 1)], dim=2)[:, None]       # (B,1,Lq,Lk+1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out)
